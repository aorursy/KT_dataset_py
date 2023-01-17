# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import seaborn as sns

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load the data set

bike_share = pd.read_csv('../input/london-bike-sharing-dataset/london_merged.csv')

bike_share.head()
bike_share['season']= bike_share['season'].map({0:'spring', 1:'summer', 2:'autumn', 3:'winter'})

bike_share['weather'] = bike_share['weather_code'].map({1:'Clear', 2:'Few Clouds', 3:'Broken Clouds', 4:'Cloudy', 7:'Rain', 10:'Stormy', 26:'Snow', 94:'Freezing Fog'})

bike_share = bike_share.drop(['weather_code'],axis=1)



#write a summary of the data 

bike_share.describe()
bike_share['timestamp'] = pd.to_datetime(bike_share['timestamp'])

bike_share['year'] = bike_share['timestamp'].dt.year

bike_share['month'] = bike_share['timestamp'].dt.month

bike_share['day'] = bike_share['timestamp'].dt.day

bike_share['weekday'] = bike_share['timestamp'].dt.weekday_name
#check for NaN values

bike_share.isnull().sum()
# Set up the matplotlib figure

f, ax = plt.subplots(2,2,figsize=(15, 12))



sns.barplot(x='weather',y='cnt', data=bike_share, order=['Clear','Few Clouds', 'Broken Clouds', 'Cloudy','Rain', 'Stormy', 'Snow','Freezing Fog'], ax=ax[0][0])

sns.barplot(x='season',y='cnt', data=bike_share, ax=ax[0][1])

sns.barplot(x='is_weekend',y='cnt', data=bike_share, ax=ax[1][0]) 

sns.barplot(x='is_holiday',y='cnt', data=bike_share, ax=ax[1][1]) 



for i in range(len(ax)):

    for k in range(2):

        #ax[i].tick_params(axis="x", labelsize=12, rotation=45 ) 

        plt.setp(ax[i][k].xaxis.get_majorticklabels(), rotation=45, ha="right", size = 12)

        plt.setp(ax[i][k].yaxis.get_majorticklabels(), size = 12)

        ax[i][k].xaxis.label.set_fontsize(20)

        ax[i][k].yaxis.label.set_fontsize(20)



        plt.tight_layout()


#f, ax = plt.subplots(1,figsize=(15, 12))

f, ax = plt.subplots(2,2,figsize=(15, 12))

sns.lineplot(data=bike_share,x='hum', y='cnt',ax=ax[0][0])

sns.lineplot(data=bike_share,x='wind_speed', y='cnt',ax=ax[0][1])

sns.lineplot(data=bike_share,x='t1', y='cnt',ax=ax[1][0])

sns.lineplot(data=bike_share,x='t2', y='cnt',ax=ax[1][1])



for i in range(len(ax)):

    for k in range(2):

        #ax[i].tick_params(axis="x", labelsize=12, rotation=45 ) 

        plt.setp(ax[i][k].xaxis.get_majorticklabels(), size = 12)

        plt.setp(ax[i][k].yaxis.get_majorticklabels(), size = 12)

        ax[i][k].xaxis.label.set_fontsize(20)

        ax[i][k].yaxis.label.set_fontsize(20)

plt.tight_layout()
correlations = bike_share.drop(['month', 'day', 'year', 'is_holiday', 'is_weekend'], axis =1).corr()

sns.heatmap(correlations, annot = True)
f, ax = plt.subplots(figsize=(20, 12))

sns.lineplot(bike_share['timestamp'], bike_share['cnt'] )



plt.setp(ax.xaxis.get_majorticklabels(), size = 12)

plt.setp(ax.yaxis.get_majorticklabels(), size = 12)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)

plt.tight_layout()
f, ax = plt.subplots(1,2,figsize=(20, 12))

sns.boxplot(bike_share['weekday'], bike_share['cnt'], ax = ax[0])

sns.boxplot(bike_share['month'], bike_share['cnt'], ax = ax[1] )



for i in range(len(ax)):

        #ax[i].tick_params(axis="x", labelsize=12, rotation=45 ) 

        plt.setp(ax[i].xaxis.get_majorticklabels(), size = 12)

        plt.setp(ax[i].yaxis.get_majorticklabels(), size = 12)

        ax[i].xaxis.label.set_fontsize(20)

        ax[i].yaxis.label.set_fontsize(20)

plt.tight_layout()