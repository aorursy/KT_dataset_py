import numpy as np 

import pandas as pd

import seaborn as sns

import sklearn 

import matplotlib.pyplot as plt

%matplotlib inline

import os

os.chdir('../input')
df = pd.read_csv('melb_data.csv')

df.head()
df
df.info()
print(df.select_dtypes(['object']).columns)
obj_cats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea','Regionname']



for colname in obj_cats:

    df[colname] = df[colname].astype('category')  

## # Convert objects to categorical variables
## Convert to date object

df['Date'] = pd.to_datetime(df['Date'])
df.describe().transpose()
num_cats = ['Postcode']  



for colname in num_cats:

    df[colname] = df[colname].astype('category')   





df.info()
df['Rooms v Bedroom2'] = df['Rooms'] - df['Bedroom2']

## this is done to remove the data multiplication of rooms, since both the values are present in 'Rooms' and in 'bedroom2'

df
df = df.drop(['Bedroom2','Rooms v Bedroom2'],1)

## The differences between these variables are minimal so keeping both would only be duplicating information. 

##Thus, the Bedroom2 feature will be removed from the data set altogether to allow for better analysis downstream.
df['Age'] = 2017 - df['YearBuilt']



# Identify historic homes

df['Historic'] = np.where(df['Age']>=50,'Historic','Contemporary')



# Convert to Category

df['Historic'] = df['Historic'].astype('category')

df.info()
# Count of missing values

df.isnull().sum()
## remove missing value

df= df.dropna()

df.info()
sns.regplot(x='Rooms',y='Price',data=df)
plt.plot(df['Price'])
plt.figure(figsize=(18,5))

sns.countplot(df['CouncilArea'])

corr=df.corr()

corr = (corr)

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)



plt.scatter(x=df['Bathroom'], y=df['Price'])

plt.ylabel('House Price')

plt.xlabel('Bathroom')

plt.show()
plt.scatter(x=df['Distance'], y=df['Price'])

plt.ylabel('House Price')

plt.xlabel('Distance from the main place')

plt.show()
plt.scatter(x=df['Car'], y=df['Price'])

plt.ylabel('House Price')

plt.xlabel('Number of carspots')

plt.show()
plt.scatter(x=df['Landsize'], y=df['Price'])

plt.ylabel('House Price')

plt.xlabel('Land size')

plt.show()
plt.scatter(x=df['BuildingArea'], y=df['Price'])

plt.ylabel('House Price')

plt.xlabel('Size of the building')

plt.show()
sns.regplot(x='Propertycount',y='Price',data=df)
plt.figure(figsize=(18,5))

sns.countplot(df['Propertycount'])

sns.regplot(x='Age',y='Price',data=df)
plt.scatter(x=df['YearBuilt'], y=df['Price'])

plt.ylabel('House Price')

plt.xlabel('YearBuilt')

plt.show()