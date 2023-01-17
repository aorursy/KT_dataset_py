#import pandas library

import pandas as pd

import numpy as np
#read data 

df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
#Show dataframe attributes 

columns = df_train.columns

columns
index = df_train.index

index
df_train.head()
#choose column and show statistic

df_train['SalePrice'].describe()
#First import seaborn and 

import seaborn as sns

import matplotlib.pyplot as plt
#create histogram

plt.figure(figsize=(15,10))

ax = sns.distplot(df_train['SalePrice'], rug =True)

ax.set_title('sdsd')
#find relation garage capacity impact saleprice

ax = plt.subplot()

ax = sns.boxplot(x = df_train['GarageCars'] , y = 'SalePrice' , data = df_train  )
#Kitchen: Number of kitchens

ax= plt.subplot()

ax = sns.boxplot(x = df_train['KitchenAbvGr'], y  = 'SalePrice' , data = df_train)
#scatter plot area with sale price 

plt.figure(figsize=(15,10))

ax = sns.regplot(x = 'GrLivArea' , y = 'SalePrice', data =df_train)

ax.set_title('Space impact house price?')

ax.set_xlabel('AREA')

ax.set_ylabel('SALE PRICE')
pair_grid = sns.pairplot(df_train[['GrLivArea', 'SalePrice']])
#Check missing data use method .isnull()

total = df_train.isnull().sum().sort_values(ascending = False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending= False)
#show result  total and percent missing data



missing_data = pd.concat([total,percent], axis =1, keys=['Total','Percent'])

missing_data.head()
ax = sns.regplot(x = 'GrLivArea' , y = 'SalePrice', data =df_train)

ax.set_title('Space impact house price?')

ax.set_xlabel('AREA')

ax.set_ylabel('SALE PRICE')
ax= plt.subplot()

ax = sns.boxplot(x = df_train['KitchenAbvGr'], y  = 'SalePrice' , data = df_train)

ax.set_xlabel('Kitchen')

ax.set_ylabel('SALE PRICE')