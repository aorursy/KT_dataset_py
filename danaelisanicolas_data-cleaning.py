import numpy as np

import pandas as pd



import datetime

import math
!ls ../input
rain = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')

rain.head()
#drop risk_mm

rain.drop(['RISK_MM'], axis=1, inplace=True)

rain.head()
for col in rain.columns:

    print(col + ' has ' + str(round((rain[col].isnull().sum() / rain.shape[0]) * 100, 2)) + '% missing values')
rain.info()
#set date to datetime object

rain['Date'] = pd.to_datetime(rain['Date'])

rain.head()
#get unique locations

locations = rain['Location'].unique()

locations
rain.drop(['Evaporation', 'Sunshine'], axis=1, inplace=True)
month = [d.month for d in rain['Date']]

rain['Month'] = month
rain.head()
def compute_missing_seasonal_num_values(dataframe, column_name):

    #separate to seasons

    defaults = {'WindGustSpeed': 5.4, 'Pressure9am': 1013.00, 'Pressure3pm': 1013.00}

    

    spring = dataframe[(dataframe['Month']) >= 9 & (dataframe['Month'] < 12)]

    sp_mean = np.mean(spring.loc[:,column_name])

    if (math.isnan(sp_mean) == True) | (np.isnan(sp_mean) == True):

        sp_mean = defaults[column_name]

    

    summer = dataframe[((dataframe['Month'] >= 1) | (dataframe['Month'] < 3)) & (dataframe['Month'] == 12)]

    sm_mean = np.mean(summer.loc[:,column_name])

    if (math.isnan(sm_mean) == True) | (np.isnan(sm_mean) == True):

        sm_mean = defaults[column_name]

    

    fall = dataframe[(dataframe['Month'] >= 3) & (dataframe['Month'] < 6)]

    fa_mean = np.mean(fall.loc[:,column_name])

    if (math.isnan(fa_mean) == True) | (np.isnan(fa_mean) == True):

        fa_mean = defaults[column_name]



    winter = dataframe[(dataframe['Month'] >= 6) & (dataframe['Month'] < 9)]

    wt_mean = np.mean(winter.loc[:,column_name])

    if (math.isnan(wt_mean) == True) | (np.isnan(wt_mean) == True):

        wt_mean = defaults[column_name]



    return sp_mean, sm_mean, fa_mean, wt_mean
def fill_missing_seasonal_num_values(dataframe, location, column_name):

    dfs = []

        

    sp, sm, fa, wt = compute_missing_seasonal_num_values(dataframe[dataframe['Location'] == location], column_name)

    df = dataframe[dataframe['Location'] == location]



    sp_df = df[(df['Month'] >= 9) & (df['Month'] < 12)]

    sp_df[column_name].fillna(sp, inplace=True)

    

    sm_df = df[((df['Month'] >= 1) & (df['Month'] < 3)) | (df['Month'] == 12)]

    sm_df[column_name].fillna(sm, inplace=True)

    

    fa_df = df[(df['Month'] >= 3) & (df['Month'] < 6)]

    fa_df[column_name].fillna(fa, inplace=True)

    

    wt_df = df[(df['Month'] >= 6) & (df['Month'] < 9)]

    wt_df[column_name].fillna(wt, inplace=True)



    dfs.append(sm_df)

    dfs.append(fa_df)

    dfs.append(wt_df)



    df = pd.concat(dfs)

        

    return df
cols = ['MinTemp', 'MaxTemp', 'Temp9am', 'Temp3pm', 'Humidity9am', 'Humidity3pm',

        'Pressure9am', 'Pressure3pm', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm']

df = rain.copy()

for col in cols:

    dfs=[]

    for location in locations:

        dfs.append(fill_missing_seasonal_num_values(df.copy(), location, col))

    df = pd.concat(dfs)



df.head()
df.isnull().sum()
rain = df

rain.isnull().sum()
rain['Rainfall'].fillna(0, inplace=True)

rain.isnull().sum()
rt_mode = rain['RainToday'].describe().top

rain['RainToday'].fillna(rt_mode, inplace=True)

rain.isnull().sum()
rain['Cloud9am'].fillna(0, inplace=True)

rain['Cloud3pm'].fillna(0, inplace=True)
def compute_missing_seasonal_cat_values(dataframe, column_name):

    #separate to seasons

    

    spring = dataframe[(dataframe['Month']) >= 9 & (dataframe['Month'] < 12)]

    sp_mode = spring[column_name].describe().top

    if (type(sp_mode) == float):

        sp_mode = dataframe[column_name].describe().top

    

    summer = dataframe[((dataframe['Month'] >= 1) | (dataframe['Month'] < 3)) & (dataframe['Month'] == 12)]

    sm_mode = summer[column_name].describe().top

    if (type(sm_mode) == float):

        sm_mode = dataframe[column_name].describe().top



    fall = dataframe[(dataframe['Month'] >= 3) & (dataframe['Month'] < 6)]

    fa_mode = fall[column_name].describe().top

    if (type(fa_mode) == float):

        fa_mode = dataframe[column_name].describe().top



    winter = dataframe[(dataframe['Month'] >= 6) & (dataframe['Month'] < 9)]

    wt_mode = winter[column_name].describe().top

    if (type(wt_mode) == float):

        wt_mode = dataframe[column_name].describe().top

       

    return sp_mode, sm_mode, fa_mode, wt_mode
def fill_missing_seasonal_cat_values(dataframe, location, column_name):

    dfs = []

        

    sp, sm, fa, wt = compute_missing_seasonal_cat_values(dataframe[dataframe['Location'] == location], column_name)

    df = dataframe[dataframe['Location'] == location]



    sp_df = df[(df['Month'] >= 9) & (df['Month'] < 12)]

    sp_df[column_name].fillna(sp, inplace=True)

    

    sm_df = df[((df['Month'] >= 1) & (df['Month'] < 3)) | (df['Month'] == 12)]

    sm_df[column_name].fillna(sm, inplace=True)

    

    fa_df = df[(df['Month'] >= 3) & (df['Month'] < 6)]

    fa_df[column_name].fillna(fa, inplace=True)

    

    wt_df = df[(df['Month'] >= 6) & (df['Month'] < 9)]

    wt_df[column_name].fillna(wt, inplace=True)



    dfs.append(sp_df)

    dfs.append(sm_df)

    dfs.append(fa_df)

    dfs.append(wt_df)



    df = pd.concat(dfs)

        

    return df
cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']

df = rain.copy()

for col in cols:

    dfs=[]

    for location in locations:

        dfs.append(fill_missing_seasonal_cat_values(df.copy(), location, col))

    df = pd.concat(dfs)



df.head()
top = df['WindGustDir'].describe().top

df['WindGustDir'].fillna(top, inplace=True)

df.isnull().sum()
rain = df
rain.isnull().sum()
rain.to_csv('cleaned_weatherAUS.csv')