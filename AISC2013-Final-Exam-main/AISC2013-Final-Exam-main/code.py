#!/usr/bin/env python
# coding: utf-8

# # Group Members
# 
# Supraja Kadaru (500197632)
# 
# Ashwitha Annapureddy (500199907)
# 
# Sai Kalyan Vollala (500197007)
# 
# Anshu Saggar (500199448)
# 

# In[16]:


# Importing pandas, numpy, matplotlib, seaborn using import and setting matplotlib inline function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[17]:


#Reading the given csv file
ad_data = pd.read_csv("advertising.csv")


# In[18]:


#Displaying the head of file i.e., we are displaying the first 5 rows of advertising
ad_data.head()


# In[19]:


#displayig the information of csv file
# info has type of index, data types of columns  etc
ad_data.info()


# In[20]:


#displaying the statistical data of file using describe function
ad_data.describe()


# In[21]:


#Plotting the histogram of Age column of csv file using matplotlib
import matplotlib.pyplot as plt
x = ad_data["Age"]
plt.hist(x, color = "Orange" ,bins = 70)
plt.grid()
plt.show()


# In[22]:


#creating a joint plot of Area income vs age using seaborn
sns.set_palette("plasma_r")
sns.set_style("whitegrid")
#importing scipy.stats and trying to plot graph between  area income vs age 
#using joint plot in seaborn-
import scipy.stats as stats
graph = sns.jointplot(x="Age",y="Area Income", data = ad_data)
x=ad_data["Age"]
y=ad_data["Area Income"]
r,p =stats.pearsonr(x , y)
# if we want to write your own legend, then we should adjust the properties then
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
# here graph is not a ax but a joint grid, so we access the axis through ax_joint method

graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])


# In[23]:


#creating a joint plot of daily time spent on site vs age using seaborn
sns.set_palette("PRGn_r")
sns.set_style("darkgrid")
#importing scipy.stats and trying to plot graph between  daily time spent on site vs age 
#using joint plot in seaborn
import scipy.stats as stats
graph = sns.jointplot(x="Age",y="Daily Time Spent on Site", kind = "kde", data = ad_data)
x=ad_data["Age"]
y=ad_data["Daily Time Spent on Site"]
r,p =stats.pearsonr(x , y)
# if we want to write your own legend, then we should adjust the properties then
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
# here graph is not a ax but a joint grid, so we access the axis through ax_joint method

graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])


# In[24]:


#creating a joint plot of daily time spent on site vs daily internet usage using seaborn 
sns.set_palette("Oranges_r")
sns.set_style("darkgrid")
#importing scipy.stats and trying to plot graph between  daily time spent on site vs daily internet usage 
#using joint plot in seaborn- Supraja Kadaru
import scipy.stats as stats
graph = sns.jointplot(x="Daily Time Spent on Site",y= "Daily Internet Usage", data = ad_data)
y=ad_data["Daily Internet Usage"]
x=ad_data["Daily Time Spent on Site"]
r,p =stats.pearsonr(x , y)
# if we want to write your own legend, then we should adjust the properties then
phantom, = graph.ax_joint.plot([], [], linestyle="", alpha=0)
# here graph is not a ax but a joint grid, so we access the axis through ax_joint method

graph.ax_joint.legend([phantom],['r={:f}, p={:f}'.format(r,p)])


# In[25]:


#creating a pairplot with the hue defined by the clicked on Ad column feature
sns.pairplot(ad_data, hue = "Clicked on Ad")
sns.set_palette("twilight_r")


# In[26]:


#splitting data into training and testing sets using train_test_split
from sklearn.model_selection import train_test_split


# In[27]:


x = ad_data[["Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Male"]]
y = ad_data["Clicked on Ad"]
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size = 0.2, random_state = 15)


# In[28]:


#training and fitting a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lm = LogisticRegression()
lm.fit(x_train, y_train)


# In[29]:


#predicting values for testing data and printing accuracy for check
y_pred = lm.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lm.score(x_test, y_test)))


# In[30]:


#Creating classification report for model
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




# In[ ]:




