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

gas = pd.read_csv("gas_cleaned.csv")
gas.info()
gas.head()
gas.shape
#show types of the values 

gas.dtypes
#change to DateTime format

gas["timestamp"] = pd.to_datetime(gas["timestamp"], format = "%Y-%m-%d %H:%M:%S")
#show types of the values 

#check that changing to DateTime format worked

gas.dtypes
#checked for misssing values 

gas.isnull().sum()
#to visualize missing values 

msno.matrix(gas)
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
missing_data(gas)
temp = missing_data(gas)

col_names = temp.T.query('Percentage_of_Missing_Values > 0.5').index
gas[col_names]
#removed the columns/locations with more than 50% missing values 

gas_cleaned = gas.drop(gas[col_names], axis = 1)
gas_cleaned.head()
gas_cleaned.shape
#to visualize missing values 

msno.matrix(gas_cleaned)
#interpolate 

gas_cleaned = gas_cleaned.interpolate(method='slinear')
gas_cleaned.isnull().sum()
#to visualize missing values 

msno.matrix(gas_cleaned)
#Last column with missing values 

gas_cleaned["Panther_education_Teofila"].isnull().sum()
#shows the number of non-zero values per column 

gas_cleaned.loc[:, gas_cleaned.columns != 'timestamp'].astype(bool).sum(axis=0)
#back propagation fill of Panther_education_Teofila 

gas_cleaned = gas_cleaned.fillna('bfill')
#to visualize missing values 

msno.matrix(gas_cleaned)
#save as csv

gas_cleaned.to_csv('/kaggle/working/gas_cleaned_new.csv')