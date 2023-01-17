# Import the modules



import pandas as pd

import numpy as np

from scipy import stats

import sklearn as sk

import itertools

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

sns.set(style='white', context='notebook', palette='deep') 



from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split 

from sklearn.model_selection import GridSearchCV

from sklearn import metrics
data = pd.read_csv('../input/nyc-rolling-sales.csv')
data.head()
print(data.isnull().sum())
data.dtypes
data['Unnamed: 0'].value_counts()
#Delete the column

data = data.drop('Unnamed: 0', axis=1)
len(data)
#Get the names of each column. I need to check that for duplicates across all columns

columns = data.columns

#Count the number of duplicates

sum(data.duplicated(columns))
#Delete the duplicates and check that it worked

data = data.drop_duplicates(columns, keep='last')

sum(data.duplicated(columns))
data['BOROUGH'].value_counts()
data['BOROUGH'][data['BOROUGH'] == 1] = 'Manhatten'

data['BOROUGH'][data['BOROUGH'] == 2] = 'Bronx'

data['BOROUGH'][data['BOROUGH'] == 3] = 'Brooklyn'

data['BOROUGH'][data['BOROUGH'] == 4] = 'Queens'

data['BOROUGH'][data['BOROUGH'] == 5] = 'Staten Island'
data.head()
sns.countplot(y = 'BOROUGH',

              data = data,

              order = data['BOROUGH'].value_counts().index)

plt.show()
data['NEIGHBORHOOD'].value_counts()
sns.countplot(y = 'NEIGHBORHOOD',

              data = data,

              order = data['NEIGHBORHOOD'].value_counts().index)

plt.show()
data['TAX CLASS AT PRESENT'].value_counts()
(len(data[data['TAX CLASS AT PRESENT']=='1'])+len(data[data['TAX CLASS AT PRESENT']=='1']))/len(data)*100
len(data[data['TAX CLASS AT PRESENT']==' '])/len(data)*100
count = data.groupby(data['TAX CLASS AT PRESENT']).count()

count = pd.DataFrame(count.to_records())

count = count.sort_values(by= 'BOROUGH', ascending = False)

count = count['TAX CLASS AT PRESENT']



sns.countplot(y='TAX CLASS AT PRESENT', data=data, order=count)
data['BBL'] = data['BOROUGH'] + '_' + data['BLOCK'].astype(str) + '_' + data['LOT'].astype(str)
data['BBL'].value_counts()
data['EASE-MENT'].value_counts()
data['BUILDING CLASS CATEGORY'].value_counts()
data['BUILDING CLASS AT PRESENT'].value_counts()
data.pivot_table(index='BUILDING CLASS AT PRESENT', columns='BUILDING CLASS CATEGORY', aggfunc='count')
data['BUILDING CLASS AT TIME OF SALE'].value_counts()
data.pivot_table(index='BUILDING CLASS AT TIME OF SALE', columns='BUILDING CLASS CATEGORY', aggfunc='count')
data['ADDRESS'].value_counts()
#This counts those addresses that contain apartment numbers or sth else - what about those roqgue letters?

len(data[data['ADDRESS'].str.contains(',')])
data['APARTMENT NUMBER'].value_counts()
len(data[data['APARTMENT NUMBER']==' '])
len(data[data['APARTMENT NUMBER']==' '])/len(data)*100
data['ZIP CODE'].value_counts()
data = data.drop(data[data['ZIP CODE']==0].index)
fig, ax = plt.subplots(figsize=(15,8)) 

sns.boxplot(x='ZIP CODE', y='BOROUGH', data=data, ax=ax)
data['RESIDENTIAL UNITS'].value_counts()
data['RESIDENTIAL UNITS'].describe()
fig, ax = plt.subplots(figsize=(10,5)) 

sns.boxplot(x='RESIDENTIAL UNITS', data=data, ax=ax)
data['COMMERCIAL UNITS'].value_counts()
len(data[data['COMMERCIAL UNITS']==0])/len(data)*100
data['COMMERCIAL UNITS'].describe()
fig, ax = plt.subplots(figsize=(10,5)) 

sns.boxplot(x='COMMERCIAL UNITS', data=data)
data['TOTAL UNITS'].value_counts()
data['TOTAL UNITS'].describe()
fig, ax = plt.subplots(figsize=(10,5)) 

sns.boxplot(x='TOTAL UNITS', data=data)
pd.set_option('display.max_columns', None)

data[data['TOTAL UNITS']==0].head(50)
pd.set_option('display.max_columns', None)

data[data['TOTAL UNITS']==0].tail(50)
sum(data['RESIDENTIAL UNITS'] + data['COMMERCIAL UNITS'] == data['TOTAL UNITS'])
sum(data['RESIDENTIAL UNITS'] + data['COMMERCIAL UNITS'] != data['TOTAL UNITS'])
data[['RESIDENTIAL UNITS','COMMERCIAL UNITS', 'TOTAL UNITS']][data['RESIDENTIAL UNITS'] + data['COMMERCIAL UNITS'] != data['TOTAL UNITS']]
data['SALE PRICE'].value_counts()
pd.set_option('display.max_columns', None)

data[data['SALE PRICE'] == ' -  '].head(50)
data['LAND SQUARE FEET'].value_counts()
(len(data[data['LAND SQUARE FEET'] == ' -  ']) + len(data[data['LAND SQUARE FEET'] == '0']))/len(data)*100
data['GROSS SQUARE FEET'].value_counts()
(len(data[data['GROSS SQUARE FEET'] == ' -  ']) + len(data[data['GROSS SQUARE FEET'] == '0']))/len(data)*100
# Are land sq feet = 0 or - the same when gross = 0 or -?
len(data[(data['GROSS SQUARE FEET'] == ' -  ') | (data['GROSS SQUARE FEET'] == '0') |

    (data['LAND SQUARE FEET'] == ' -  ') | (data['LAND SQUARE FEET'] == '0')]

    )/len(data)*100
data['YEAR BUILT'].value_counts()
fig, ax = plt.subplots(figsize=(10,5)) 

sns.boxplot(x = 'YEAR BUILT', data=data, ax=ax)
data['YEAR BUILT'].describe()
data['YEAR BUILT'][data['YEAR BUILT'] < 1750].count()
fig, ax = plt.subplots(figsize=(10,5)) 

sns.boxplot(x = 'YEAR BUILT', data=data[data['YEAR BUILT'] < 1750], ax=ax)
#Create new field: Built x years ago from 2017

data['Building_Age'] = 2017 - data['YEAR BUILT']
data['SALE DATE'].value_counts()
data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])
data['SALE DATE'].value_counts()
data['SALE YEAR'], data['SALE MONTH'], data['SALE QUARTER'] = data['SALE DATE'].dt.year, data['SALE DATE'].dt.month, data['SALE DATE'].dt.quarter
sns.boxplot(x='SALE MONTH', data=data[data['SALE YEAR']==2017])
sns.boxplot(x='SALE MONTH', data=data[data['SALE YEAR']==2016])