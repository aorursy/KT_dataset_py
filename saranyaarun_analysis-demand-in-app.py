#import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
#Read the dataset

google_data = pd.read_csv("../input/googleplaystore.csv")

#if your data set in local

#google_data = pd.read_csv("D:\Data-Analyst\googleplaystore.csv")
#Check the shape of file

google_data.shape
#Have a look on the dataset 

google_data.head(5)
#Complete Information about the dataset

google_data.info()
#Drop last 3 columns

new_data = google_data.drop(columns = ['Last Updated','Current Ver','Android Ver','Price'])
#verifying dataset after deleting the columns

new_data.info()
#Finding the wrong records as it has wrong values of Category,Reviews,size.

print(new_data[new_data['Reviews'].str.contains('3.0M')])
#drop the column by id

new_data = new_data.drop([10472])
#check data shape

new_data.shape
#convert data type for Reviews

new_data['Reviews']=new_data['Reviews'].astype(int)
#check the null values

new_data.Reviews.isnull().sum()
#Shape of the Data

print(new_data.Installs.shape[0])

print(new_data.Installs.dtypes)
#Removing the (+ & ,) from Installs data set

new_data['Installs'] = new_data['Installs'].apply(lambda x: x.strip('+'))

new_data['Installs'] = new_data['Installs'].apply(lambda x: x.replace(',',''))

new_data.head()

# check unique values

new_data['Installs'].unique()

# Installs column has 'Free' value

new_data[new_data['Installs'].str.contains('Free')]

#convert Installs column to integer as it has numeric values

new_data['Installs'] = new_data['Installs'].astype(int)

#check data type

new_data.Installs.dtypes
#check the null values from installs

new_data.Installs.isnull().sum()
#check unique values from Type

print(new_data.Type.unique())
# show the null value record

new_data[new_data['Type'].isnull()]

# since the type(nan) row has null in rating,intalls,reviews. SO drop this row

new_data = new_data.drop([9148])

#check dropped data

new_data.Type.unique()
#check the information of dataset

new_data.info()
#Find the Shape of Rating

print(new_data.Rating.shape[0])

#Find total number of nan va;ues

print(new_data.Rating.isnull().sum())

#Find missing value from Rating

new_data[new_data['Rating'].isnull()]['Installs'].mean()
#Finding the null values in Rating

new_data['Rating'].isnull().sum()
#Finding the different values in Rating column

new_data['Rating'].unique()
#Since Rating dataset has Nan values because the apps has been uploaded 

new_data2 = new_data.Rating.mean()

new_data2
#Replace the null values by mean of Rating.

new_data['Rating'] = new_data['Rating'].fillna(4.19)
#Final information about dataset

new_data.info()
#1. Studing the dataset and analyzing the Type of Apps demand in market.

new_data.Type.value_counts()
new_data.Category.value_counts().head(10)