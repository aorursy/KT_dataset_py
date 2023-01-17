#importing the required packages



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()
# importing the csv file 

data=pd.read_csv('../input/Cosmetics sale.csv')



# shape of the dataset 

data.shape
# first 5 records 

data.head(3)
# last 5 records 

data.tail(3)
data.rename(columns={'Site Id':'Warehouse',

                     'Category Name ID':'Category',

                     'Qty':'Quantity'},inplace=True)

data.columns
# technical summary of the data (column datatypes ,null value vount and the memory usage)

#data.dtypes

data.info()
data['Quantity']=[int(i)for i in data['Quantity']]

print(data['Quantity'].head(2))
data['Date']=pd.to_datetime(data['Date'])

print(data['Date'].head(2))
# finding null values if any

#pd.isnull(data).sum()

print('missing values in the data {}'.format(data.isnull().sum()))

print(data.shape)
# any records contiains missing values it shall be dropped

data.dropna(how='any',inplace=True)

print(data.shape)
data.describe(include=[np.object])
data.drop(columns=['Unit'],axis=1,inplace=True)
data.drop(columns=['Size'],axis=1,inplace=True)
sns.boxplot(data['Price'],orient='vertical')
print('data shape before droping outliers :\n {}'.format(data.shape))
x=data[data['Price']>650].index

data.drop(index=x ,axis=1 ,inplace=True)



print('data shape after droping outliers :\n {}'.format(data.shape))
y=data[data['Pack Unit Id']=='empty'].index

data.drop(index=y, axis=0 ,inplace=True)

data.shape
z=data[data['Pack Unit Id']=='no'].index

data.drop(index= z, axis=0 ,inplace=True)

data.shape
data['Pack Unit Id']=data['Pack Unit Id'].str.lower()
print(data.shape)

data.head(2)
sns.kdeplot(shade=True,data=data['Price'])



print('mean of the price distribution:{}'.format(data['Price'].mean()))

print('median of the price distribution:{}'.format(data['Price'].median()))
data['Price'].mean()
data['Category'].value_counts(normalize=True)*100
data['Pack Unit Id'].value_counts(normalize=True)*100
#data.groupby('ParentSKU')['Quantity'].count().sort_values(ascending=False)

sns.set_style('white')

sns.set_context('paper',font_scale=1)

count_of_sales=data.groupby('ParentSKU')['Quantity'].count().sort_values(ascending=False)

count_of_sales.plot(figsize=(15,8),kind='bar',title='count of sales of each product')
data.groupby('ParentSKU')['Net Sales calculated'].sum().sort_values(ascending=False)

sns.set_style('white')

sns.set_context('paper',font_scale=1)

sum_of_sales=data.groupby('ParentSKU')['Net Sales calculated'].sum().sort_values(ascending=False)

sum_of_sales.plot(figsize=(15,8),kind='bar',title='sum of sales of each product')
plt.figure(figsize=(15,8))

sns.set_style('white')

sns.set_context('paper',font_scale=1)

chart=sns.lineplot(x='Warehouse',y='Net Sales calculated',data=data)

chart.set_xticklabels(labels=data['Warehouse'].unique(),rotation=90)

data['Zone']=data['Zone'].str.upper().str.replace('NORTH 1','NORTH1').str.replace('NORTH 2','NORTH2')

plt.figure(figsize=(20,8))

sns.barplot(data=data,x='Zone',y='Net Sales calculated',color='red',label='big').set_title('total Sales generated from each zone')
sns.jointplot(x='Rank',y='Price',data=data,kind='reg')