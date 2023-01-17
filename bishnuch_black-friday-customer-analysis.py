# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualizing data
import seaborn as sns 
from collections import Counter
%matplotlib inline
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/BlackFriday.csv')
#shape of the dateset (column,rows)
df.shape

#description of dataset
df.describe()
#first 5 items
df.head()
#Gender wise purchase made
def plot(group,column,plot):
    ax=plt.figure(figsize=(7,7))
    df.groupby(group)[column].sum().sort_values().plot(plot)
    
plot('Gender','Purchase','bar')
explode = (0.2,0)  
fig1, ax1 = plt.subplots(figsize=(7,4))
ax1.pie(df['Gender'].value_counts(), explode=explode,labels=['Male','Female'], autopct='%1.0f%%',
         startangle=180)
ax1.axis('equal')  
plt.legend()
plt.show()
#age wise customers distribution of male and female
#male and female customers age 26-35 who did most of the shopping
fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['Gender'],hue=df['Age'])
#overall general age group who visit the store is from 26-35
plot('Age','Purchase','bar')
#city cat A has more purchasers
explode = (0.2, 0.2, 0.2)
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.pie(df['City_Category'].value_counts(),explode=explode, labels=df['City_Category'].unique(), autopct='%1.0f%%',
         startangle=180)
ax1.axis('equal')  
plt.legend()
plt.show()
explode = (0.2, 0.2, 0.2)
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.pie(df.groupby('City_Category')['Purchase'].sum(),explode=explode, labels=df['City_Category'].unique(), autopct='%1.0f%%',
         startangle=180)
ax1.axis('equal')  
plt.legend()
plt.show()
#Cat b city has more cusomter of 26-35 and least 0-17, 55 
#cat c city has 26-35 aged customer most and least 0-17 aged
#cat a city has 26-35 aged customer most and 0-17 least.
fig1, ax1 = plt.subplots(figsize=(14,6))
sns.countplot(df['City_Category'],hue=df['Age'])
explode = (0.2, 0.2)
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.pie(df['Marital_Status'].value_counts(),explode=explode, labels=['Yes','No'], autopct='%1.0f%%',
       startangle=180)
ax1.axis('equal')  
plt.legend()
plt.show()
#no of customers who hav which occupation
fig1, ax1 = plt.subplots(figsize=(14,6))
df['Occupation'].value_counts().sort_values().plot('bar')
#occupation of purchasers
plot('Occupation','Purchase','bar')
#product 1 purchasal
plot('Product_Category_1','Purchase','barh')
##product 2 purchasal
plot('Product_Category_2','Purchase','barh')
#product 3 purchasal
plot('Product_Category_3','Purchase','barh')
#most purchased product top 3
fig1, ax1 = plt.subplots(figsize=(14,6))
df.groupby('Product_ID')['Purchase'].count().nlargest(3).sort_values().plot('barh')
#least purchased product top 3
fig1, ax1 = plt.subplots(figsize=(14,6))
df.groupby('Product_ID')['Purchase'].count().nsmallest(3).sort_values().plot('barh')
#customers staying in current cities made purchases
labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']
explode = (0.2, 0.2, 0.2 ,0.2 ,0.2)
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.pie(df.groupby('Stay_In_Current_City_Years')['Purchase'].sum(),explode=explode, labels=labels, autopct='%1.0f%%',
     startangle=180)
ax1.axis('equal')  
plt.legend()
plt.show()
#customers staying in current cities visited the store
labels=['First Year','Second Year','Third Year','More Than Four Years','Geust']
explode = (0.2, 0.2, 0.2, 0.2, 0.2)
fig1, ax1 = plt.subplots(figsize=(14,6))
ax1.pie(df['Stay_In_Current_City_Years'].value_counts(),explode=explode, labels=labels, autopct='%1.0f%%',
         startangle=180)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.legend()
plt.show()
# most no of purchases done with respect to staying in current city
plot('Stay_In_Current_City_Years','Purchase','bar')