import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline

import os
os.chdir('../input')
data= pd.read_csv('melbourne.csv') 
data
data.info() 
data.shape
data.median() 
data.describe() 
missing= data.isnull().sum() 

missing = missing[missing > 0]

missing.plot.bar() 
missing
#replacing missing values of Car,BuildingArea and YearBuilt with their mean values and CouncilArea with None

data['Car'].replace({np.nan:1.6},inplace= True) 

data['BuildingArea'].replace({np.nan:152},inplace= True)

data['YearBuilt'].replace({np.nan:1964},inplace= True)

data['CouncilArea'].replace({np.nan:'None'},inplace= True) 
data.head(3) 
qwe = data.corr()

plt.figure(figsize=(14,14))

sns.heatmap(qwe, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws= {'size': 12},

            xticklabels= qwe.columns.values,

            yticklabels= qwe.columns.values, linewidths= 0.5, linecolor= 'gold') 

num= [f for f in data.columns if data[f].dtype != 'object']

print("Numerical features are {}".format(len(num)))

cat= [f for f in data.columns if data[f].dtype == 'object']

print("Categorical features are {}".format(len(cat))) 
cat
num
X1= data[['Rooms','Distance','Postcode','Bedroom2','Price']] 

sns.set(style= 'ticks', palette= 'Dark2')

sns.pairplot(data, vars= X1)

plt.show() 
sns.lmplot(x= 'Rooms', y= 'Price', data= data) 
sns.lmplot(x= 'Distance', y= 'Price', data= data)
sns.lmplot(x= 'Postcode', y= 'Price', data= data)
sns.lmplot(x= 'Bedroom2', y= 'Price', data= data)
X2= data[['Bathroom','Car','Landsize','BuildingArea','YearBuilt','Price']] 

sns.set(style= 'ticks', palette= 'icefire')

sns.pairplot(data, vars= X2)

plt.show() 
sns.lmplot(x= 'Bathroom', y= 'Price', data= data)
sns.lmplot(x= 'Car', y= 'Price', data= data) 
sns.lmplot(x= 'Landsize', y= 'Price', data= data)
sns.lmplot(x= 'BuildingArea', y= 'Price', data= data)
sns.lmplot(x= 'YearBuilt', y= 'Price', data= data)
X3= data[['Lattitude','Longtitude','Propertycount','Price']]

sns.set(style= 'ticks', palette= 'mako')

sns.pairplot(data, vars= X3)

plt.show() 
sns.lmplot(x= 'Lattitude', y= 'Price', data= data)
sns.lmplot(x= 'Longtitude', y= 'Price', data= data)
sns.lmplot(x= 'Propertycount', y= 'Price', data= data)
sns.distplot(a= data['Propertycount'], kde= False)

sns.boxplot(x= 'Price', y= 'Regionname', data= data)
sns.stripplot(x= 'Price', y= 'Regionname', data= data)
sns.stripplot(x= 'Price', y= 'CouncilArea', data= data)
sns.stripplot(x= 'Price', y= 'Method', data= data)
sns.stripplot(x= 'Price', y= 'Type', data= data)