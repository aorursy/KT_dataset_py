#Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import plotly as py

import seaborn as sns
#read Amazon Forest Fire data

data = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding = "ISO-8859-1")
#check for data types and data quality -- can also chec for memory usage

data.info()
#display first 5 records of dataframe

data.head(5)
#check the overall number of samples and features

print("Overall number of samples : ", data.shape[0])

print("Overall number of features : ", data.shape[1])
#check for null values

data.isnull().sum()
#checking unique values in the dataset

print("Unique values in year : ", list(data.year.unique()))

print("Unique values in state : ", list(data.state.unique()))

print("Unique values in month : ", list(data.month.unique()))

print("Unique values in date : ", list(data.date.unique()))
data.describe()
#import plotly.plotly as py

labels = data['year'].value_counts().index

values = data['year'].value_counts().values



colors = ['#eba796', '#96ebda']



fig = {'data' : [{'type' : 'pie',

                   'name' : "Fire per year",

                   'labels' : data['year'].value_counts().index,

                   'values' : data['year'].value_counts().values,

                   'direction' : 'clockwise',

                   'marker' : {'colors' : ['#9cc359', '#e96b5c']}}], 'layout' : {'title' : 'Fire per Year'}}

plt.figure(figsize = (15,5))

sns.swarmplot(x= 'year', y= 'number', data = data)

plt.show()


plt.figure(figsize = (25,10))

sns.barplot(x='state',y = 'number', data = data)

plt.xlabel('States')

plt.ylabel('Number of fire')

plt.show()