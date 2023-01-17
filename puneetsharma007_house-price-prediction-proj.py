# -*- coding: utf-8 -*-
"""
Created on Tue May 12 12:49:35 2020

@author: PuneetSharma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

house_data = pd.read_csv('C:\\Users\\PuneetSharma\\Desktop\\Work\\Data-Science-Python\\Sample-Dataset2\\test.csv')

print(house_data)

house_data.head()

house_data.shape

#creating a copy of data
house_data2 = house_data.copy()
print(house_data2)


#check the relationship between categorical variables by doing univariate analysis
#frequency table

pd.crosstab(index=house_data2['bathrooms'],columns='count',dropna=True)

##two way frequency table
pd.crosstab(index=house_data2['floors'],columns='count',dropna=True)
pd.crosstab(index=house_data2['bathrooms'],columns='count',dropna=True)

pd.crosstab(index=house_data2['floors'],columns=['bedrooms'],dropna=True)

pd.crosstab(index=house_data2['bedrooms'],columns='count',dropna=True)

pd.crosstab(index=house_data2['floors'],columns=['waterfront'],dropna=True)

pd.crosstab(index=house_data2['floors'],columns=['bedrooms'],normalize=True,dropna=True)


house_data2.info()

#convert price, floors & bedrooms to int datatype
house_data2['price'] = (house_data2['price']).astype(int)
house_data2['floors'] = (house_data2['floors']).astype(int)
house_data2['bedrooms'] = (house_data2['bedrooms']).astype(int)
house_data2.info()

#remove the missing values
house_data2.isna().sum()
house_data2= house_data2.dropna()
house_data2.info()
house_data2.head(5)

house_data2.shape

house_data2.corr()


plt.scatter(house_data2['yr_built'],house_data2['price'],c='red')
plt.title('Scatter plot of Price vs year ofthe cars')
plt.xlabel('yr_built')
plt.ylabel('price(Dollars)')
plt.show()

##creating histogram for yr_built

plt.hist(house_data2['yr_built'])

train_data = pd.read_csv('C:\\Users\\PuneetSharma\\Desktop\\Work\\Data-Science-Python\\Sample-Dataset2\\train.csv')
train_data.shape

train_data.describe()
train_data.head(3)
house_data.head(3)
train_data.shape,house_data.shape

## check for duplicates in IDS
#idsunique  = len(set(train_data.Id))
idsunique1  = len(train_data.Id)
print(idsunique1) 

idsTotal = train_data.shape[0]
print(idsTotal)

idsdupe = idsTotal - idsunique1

print(idsdupe)

train_data.drop(['Id'],axis=1, inplace=True)

