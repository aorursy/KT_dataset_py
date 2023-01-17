# Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

import datetime
def overview():

    data = pd.read_csv('../input/weatherww2/Summary of Weather.csv')

    # Print the first 5 lines of data

    print("First 5 lines of data \n\n")

    print(data.head())

    

    # Print data type

    print("\n\n\nDatatype\n")

    print(data.dtypes)

    

    # Print number of null values 

    print("\n\n\nNumber of null values\n")

    print(data.isnull().sum())

    

    # Print data summary

    print("\n\n\nData summary\n")

    print(data.describe())

    

    # Print data shape

    print("\n\n\nData shape\n")

    print("Data has {} rows and {} columns".format(data.shape[0], data.shape[1]))

   

    return data



data = overview()
# Dropping NaN rows 

data = data.dropna(subset = ['Snowfall', 'PRCP', 'MAX', 'MIN', 'MEA', 'SNF'])



# Dropping redundant column

data = data.drop(columns = ['PRCP'])



# Dropping NaN columns 

data = data.dropna(axis = 'columns')



# Taking a look at whats left

data.isnull().sum()
# Create a function to separate out numerical and categorical data

    ## Using this function to ensure that all non-numerical in a numerical column

    ## and non-categorical in a categorical column is annotated

def cat_variable(df):

    return list(df.select_dtypes(include = ['category', 'object']))



def num_variable(df):

    return list(df.select_dtypes(exclude = ['category', 'object']))



categorical_variable = cat_variable(data)

numerical_variable = num_variable(data)



# Create a function to process outlier data

def outlier(data):

    z = np.abs(stats.zscore(data[numerical_variable]))

    z_data = data[(z < 3).all(axis=1)] # Remove any outliers with Z-score > 3 or < -3

    return z_data



data = outlier(data)
# Removing non-numeric data and cnverting date to datetime format

data['Precip'] = pd.to_numeric(data['Precip'], errors='coerce')

data['Year'] = pd.DatetimeIndex(data['Date']).year

data.applymap(np.isreal)
print('Before cleaning: \n\n{}'.format(data.isnull().sum()))



data = data.fillna(method='ffill').fillna(method='bfill')



print('\nAfter cleaning: \n\n{}'.format(data.isnull().sum()))
# Convert year to datetime format

data = data[:][:1000] # We take in the first 1000 data since we do not want to cluster bomb the whole plot. 

data['Date'] = pd.to_datetime(data['Date'])



# Indexing the year

data = data.set_index('Date')

axes = data[["MaxTemp", "MinTemp", "MeanTemp", "Precip"]].plot(figsize=(11, 9), subplots=True, linewidth=1)