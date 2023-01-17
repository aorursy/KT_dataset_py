import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.chdir('/kaggle/input/buildingdatagenomeproject2')

os.listdir()
#imports

import pandas as pd

import numpy as np

import seaborn as sns

import missingno as msno
#load dataset 

electricity = pd.read_csv("electricity_cleaned.csv")
electricity.info()
electricity.head()
electricity.shape
#show types of the values 

electricity.dtypes
#change to DateTime format

electricity["timestamp"] = pd.to_datetime(electricity["timestamp"], format = "%Y-%m-%d %H:%M:%S")
#show types of the values 

#check that changing to DateTime format worked

electricity.dtypes
#checked for misssing values 

electricity.isnull().sum()
#to visualize missing values 

msno.matrix(electricity)
#function shows the percentage of missing values and type of the values

def missing_data(data):

    percent = (data.isnull().sum() / data.isnull().count())

    x = pd.concat([percent], axis=1, keys=['Percentage_of_Missing_Values'])

    type = []

    

    for col in data.columns:

        dtype = str(data[col].dtype)

        type.append(dtype)

    x['Data Type'] = type

    

    return(np.transpose(x))
missing_data(electricity)
temp = missing_data(electricity)

col_names = temp.T.query('Percentage_of_Missing_Values > 0.5').index
electricity[col_names]
#removed the columns/locations with more than 50% missing values 

electricity_cleaned = electricity.drop(electricity[col_names], axis = 1)
electricity_cleaned.head()
electricity_cleaned.shape
#to visualize missing values 

msno.matrix(electricity_cleaned)
#interpolate 

electricity_cleaned = electricity_cleaned.interpolate(method='slinear')
electricity_cleaned.isnull().sum()
#to visualize missing values 

msno.matrix(electricity_cleaned)
#shows the number of non-zero values per column 

electricity_cleaned.loc[:, electricity_cleaned.columns != 'timestamp'].astype(bool).sum(axis=0)
#back propagation fill

electricity_cleaned = electricity_cleaned.fillna(method='bfill')
#to visualize missing values 

msno.matrix(electricity_cleaned)
#forward propagation fill 

electricity_cleaned = electricity_cleaned.fillna(method='ffill') 
#to visualize missing values 

msno.matrix(electricity_cleaned)
electricity_cleaned.isnull().sum()
#save as csv

electricity_cleaned.to_csv('/kaggle/working/electricity_cleaned_new.csv')