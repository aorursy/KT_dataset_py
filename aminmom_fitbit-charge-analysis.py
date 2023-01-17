# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

dir_data = r'../input/'
print(os.listdir(dir_data))


from __future__ import division
from __future__ import print_function

import time

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler, scale, StandardScaler, MinMaxScaler

import tensorflow as tf
print(tf.__version__)


## plotting
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
%matplotlib inline
# Any results you write to the current directory are saved as output.


## function to import csv and do QC

def import_QC(folder,file_in,sep_file=','):
    """
    The function performs CSV file import and performs
    QC of shape and head
    
    outputs a pandas dataframe
    """
    ## Import data
    file_in = folder + '/' + file_in 
    temp_df = pd.read_csv(file_in,sep=sep_file)
    
    ## QC dataframe 
    print(temp_df.shape)
    print(temp_df.head())
    
    ## return
    return(temp_df)
## Import data training
folder = dir_data
file_in = r'One_Year_of_FitBitChargeHR_Data.csv'
tracker_df = import_QC(folder,file_in)
## Convert the date column datatype to datatime and 
## extract the day of the month

tracker_df['Date'] = pd.to_datetime(tracker_df['Date'],format='%d-%m-%Y')
tracker_df['Date_only'] = pd.to_datetime(tracker_df['Date']).dt.day
tracker_df.index = tracker_df['Date']
del tracker_df['Date']
## sum  the slow, moderate and intense activity and create 'all_activity'

tracker_df['all_activity'] = tracker_df[['Minutes_of_intense_activity','Minutes_of_moderate_activity','Minutes_of_slow_activity']].sum(axis=1)


print(tracker_df.dtypes)
tracker_df.head()
### Groupby the day of the month and make a boxplot of calories burnt

# figure size
plt.figure(figsize=(15,8))

# Usual boxplot
ax = sns.boxplot(x='Date_only', y='Calories', data=tracker_df)
 
# Add jitter with the swarmplot function.
ax = sns.swarmplot(x='Date_only', y='Calories', data=tracker_df, color="grey")

ax.set_title('Box plot of Calories with Jitter bu day of the month')
## Groupby the weekday and plot statistics by day of the week

by_weekday = pd.DataFrame()
by_weekday['Calories'] = tracker_df['Calories'].groupby(tracker_df.index.dayofweek).sum()
by_weekday['Week_day'] = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

# figure size
plt.figure(figsize=(15,8))

# simple barplot
ax = sns.barplot(x='Week_day', y='Calories',  data=by_weekday)

ax.set_title('Barplot of calories by the day of the week')
### Scatterplot of Calories Vs intense activity

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Calories', y='Minutes_of_intense_activity', data=tracker_df)

ax.set_title('Scatterplot of calories and intense_activities')
### Scatterplot of Calories Vs moderate activity

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Calories', y='Minutes_of_moderate_activity', data=tracker_df)

ax.set_title('Scatterplot of calories and moderate_activities')
### Scatterplot of Calories Vs slow activity

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Calories', y='Minutes_of_slow_activity', data=tracker_df)

ax.set_title('Scatterplot of calories and slow_activities')
### Scatterplot of Calories Vs all_activity

# figure size
plt.figure(figsize=(15,8))

# Simple scatterplot
ax = sns.scatterplot(x='Calories', y='all_activity', data=tracker_df)

ax.set_title('Scatterplot of calories and all_activities')
## plot the raw values 

col_select = ['Calories','Minutes_of_intense_activity','Minutes_of_moderate_activity','Minutes_of_slow_activity','all_activity']
wide_df = tracker_df[col_select]

# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df)

ax.set_title('Un-normalized value of calories and different activities')
## Calculate the rolling averages

calories_roll = tracker_df['Calories'].rolling(window='10D').mean()
slow_activity_roll = tracker_df['Minutes_of_slow_activity'].rolling(window='10D').mean()
moderate_activity_roll = tracker_df['Minutes_of_moderate_activity'].rolling(window='10D').mean()
intense_activity_roll = tracker_df['Minutes_of_intense_activity'].rolling(window='10D').mean()
all_activity_roll = tracker_df['all_activity'].rolling(window='10D').mean()
## Plot all the rolling averages

col_select = ['Calories','Minutes_of_intense_activity','Minutes_of_moderate_activity','Minutes_of_slow_activity','all_activity']
wide_df = pd.concat([calories_roll,slow_activity_roll,moderate_activity_roll,intense_activity_roll,all_activity_roll],axis=1)

# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df)

ax.set_title('Un-normalized rolling mean of calories and different activities')

## Plot the scaled rolling averages

wide_df = pd.concat([calories_roll,slow_activity_roll,moderate_activity_roll,intense_activity_roll,all_activity_roll],axis=1)
scaler = MinMaxScaler()

wide_df_scaled = pd.DataFrame(scaler.fit_transform(wide_df), columns=wide_df.columns)
# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df_scaled )

ax.set_title('Rolling mean of calories and different activities')
## Plot the scaled rolling averages

wide_df = pd.concat([calories_roll,intense_activity_roll],axis=1)
scaler = MinMaxScaler()

wide_df_scaled = pd.DataFrame(scaler.fit_transform(wide_df), columns=wide_df.columns)
# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df_scaled )

ax.set_title('Rolling mean of calories and intense activities')
## Plot the scaled rolling averages

wide_df = pd.concat([calories_roll,moderate_activity_roll],axis=1)
scaler = MinMaxScaler()

wide_df_scaled = pd.DataFrame(scaler.fit_transform(wide_df), columns=wide_df.columns)
# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df_scaled )

ax.set_title('Rolling mean of calories and moderate activities')
## Plot the scaled rolling averages

wide_df = pd.concat([calories_roll,slow_activity_roll],axis=1)
scaler = MinMaxScaler()

wide_df_scaled = pd.DataFrame(scaler.fit_transform(wide_df), columns=wide_df.columns)
# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df_scaled )

ax.set_title('Rolling mean of calories and slow activities')
## Plot the scaled rolling averages

wide_df = pd.concat([calories_roll,all_activity_roll],axis=1)
scaler = MinMaxScaler()

wide_df_scaled = pd.DataFrame(scaler.fit_transform(wide_df), columns=wide_df.columns)
# figure size
plt.figure(figsize=(15,8))

# timeseries plot using lineplot
ax = sns.lineplot(data=wide_df_scaled )

ax.set_title('Rolling mean of calories and all activities')
wide_df = pd.concat([calories_roll,slow_activity_roll,moderate_activity_roll,intense_activity_roll,all_activity_roll],axis=1)

f, ax = pl.subplots(figsize=(10, 8))
corr_temp = wide_df.corr()
ax = sns.heatmap(corr_temp, mask=np.zeros_like(corr_temp, dtype=np.bool), 
                 cmap=sns.diverging_palette(220, 10, as_cmap=True),
                 annot=True, square=True)

ax.set_title('Correlation between calories and different activities')
