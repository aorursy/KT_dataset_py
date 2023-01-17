import pandas as pd

import numpy as np

import os

import pandas_profiling

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
RAW_DATA_FILE = 'melb_data.csv'

# Check if notebook is run on Kaggle or locally

cwd = os.getcwd()

if cwd == '/kaggle/working':

    RAW_DATA_PATH = '../input'

else:

    RAW_DATA_PATH = os.path.join('../data','raw')
def load_data(raw_data_path, raw_data_file):

    cols_to_use = ['Rooms','Price','Method','Date','Distance','Propertycount','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude']

    df = pd.read_csv(os.path.join(raw_data_path, raw_data_file), usecols =[i for i in cols_to_use])

    df['Date'] = pd.to_datetime(df['Date'])

    return df



df = load_data(RAW_DATA_PATH, RAW_DATA_FILE)
df.head()
df.info()
df.Method.value_counts()
df.describe()
df.hist(bins=50, figsize=(20,15))
train_set, test_set = train_test_split(df, test_size=0.2, random_state=38, shuffle=True)

print(train_set.shape)

print(test_set.shape)
housing = train_set.copy()
bottom_99p = housing.Price < np.percentile(housing.Price, 99)

housing[bottom_99p].plot(kind='scatter', x='Longtitude', y='Lattitude', alpha=0.3, figsize=(12,10)

             , c='Price', cmap=plt.get_cmap('jet'), colorbar=True)
corr_matrix = housing.corr()

corr_matrix['Price'].sort_values(ascending=False)
corr_matrix['Rooms'].sort_values(ascending=False)
def feature_extraction(df):

    df['other_rooms'] = df.Rooms - df.Bathroom

    df['surface_per_room'] = df.BuildingArea/df.Rooms

    df['perc_built'] = df.BuildingArea/df.Landsize

    df['house_age'] = df.Date.dt.year - df.YearBuilt

    

feature_extraction(housing)
housing.corr()['Price'].sort_values(ascending=False)
pandas_profiling.ProfileReport(housing)