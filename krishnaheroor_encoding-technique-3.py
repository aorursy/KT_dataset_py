import pandas as pd #import pandas

import numpy as np #import numpy

from sklearn.preprocessing import LabelEncoder  #importing LabelEncoder
train = pd.read_csv('../input/bigmart-sales-data/Train.csv')
train.head()
#check the size of the dataset

print('Data has {} Number of rows'.format(train.shape[0]))

print('Data has {} Number of columns'.format(train.shape[1]))
#let's keep our categorical variables in one table

cat_data = train[['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type','Item_Outlet_Sales']]

cat_data.head()   #check the head of categorical data
#Let's start where we had left 

print(cat_data['Item_Type'].nunique())

print(cat_data['Item_Type'].unique())
fe = cat_data['Item_Type'].value_counts(ascending=True)/len(cat_data)  #count the frequency of labels

print(fe)
cat_data['Item_Type'].map(fe).head(10)  #map frequency to item type
#get the mean of target variable label wise

me = cat_data.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean()

print(me)
#get the mean of target variable label wise

cat_data['Outlet_Identifier'].map(me).head(10)
#check value counts in Outlet_Size

cat_data['Outlet_Size'].value_counts()
#Check the null values

cat_data['Outlet_Size'].isnull().sum()
#fill the null values with other category for now

cat_data['Outlet_Size'].fillna('Others',inplace = True)
#prepare a dictionary to map

size_fe = {"Small" : 0, "Medium" : 1, "High" : 2, "Others" : 3}

cat_data['Outlet_Size'].map(size_fe).head(10)
cat_data['Outlet_Location_Type'].value_counts()
location_fe = {"Tier 3" : 1, "Tier 2" : 2, "Tier 1" : 3}

cat_data['Outlet_Location_Type'].map(location_fe).head(10)
#Check last variable and do the encoding

cat_data['Outlet_Type'].value_counts()
pd.get_dummies(cat_data['Outlet_Type'],drop_first=True).head()