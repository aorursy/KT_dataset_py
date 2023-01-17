import pandas as pd #import pandas

import numpy as np #import numpy

from sklearn.preprocessing import LabelEncoder  #importing LabelEncoder

train = pd.read_csv('../input/bigmart-sales-data/Train.csv')
#check the head of dataset

train.head(5)
#check the size of the dataset

print('Data has {} Number of rows'.format(train.shape[0]))

print('Data has {} Number of columns'.format(train.shape[1]))
#check the information of the dataset

train.info()
#let's keep our categorical variables in one table

cat_data = train[['Item_Identifier','Item_Fat_Content','Item_Type','Outlet_Identifier','Outlet_Size','Outlet_Location_Type','Outlet_Type']]
cat_data.head()   #check the head of categorical data
cat_data.apply(lambda x: x.nunique()) #check the number of unique values in each column
#check the top 10 frequency in Item_Identifier

cat_data['Item_Identifier'].value_counts().head(10)
pd.get_dummies(cat_data['Item_Identifier'],drop_first=True)  #applying one hot encoding
#apply binary encoding on Item_Identifier

import category_encoders as ce                              #import category_encoders

encoder = ce.BinaryEncoder(cols=['Item_Identifier'])        #create instance of binary enocder

df_binary = encoder.fit_transform(cat_data)                 #fit and tranform on cat_data

df_binary.head(5)

#check the unique values 

cat_data['Item_Fat_Content'].unique()
low_fat = ['LF','low fat']

cat_data['Item_Fat_Content'].replace(low_fat,'Low Fat',inplace = True) #replace 'LF' and 'low fat' with 'Low Fat'

cat_data['Item_Fat_Content'].replace('reg','Regular',inplace = True)   #Replace 'reg' with regular
cat_data['Item_Fat_Content'].unique()
#Apply LabelEncoder

le = LabelEncoder()

cat_data['Item_Fat_Content_temp'] = le.fit_transform(cat_data['Item_Fat_Content'])

print(cat_data['Item_Fat_Content'].head())

print(cat_data['Item_Fat_Content_temp'].head())
#prepare a dict to map

mapping = {'Low Fat' : 0,'Regular': 1} #map Low Fat as 0 and Regular as 1

cat_data['Item_Fat_Content_temp1'] = cat_data['Item_Fat_Content'].map(mapping)

cat_data['Item_Fat_Content_temp1'].head()
factorized,index = pd.factorize(cat_data['Item_Fat_Content'])  #using pd.factorize it gives us factorized array and index values

print(factorized)

print(index)
#Let's look at item type column

print(cat_data['Item_Type'].nunique())  #check number of unique values

print(cat_data['Item_Type'].unique())   #check the unique values