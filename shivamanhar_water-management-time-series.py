# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = "../input"

reservoir_levels = pd.read_csv(path+'/chennai_reservoir_levels.csv')

reservoir_rainfall = pd.read_csv(path+'/chennai_reservoir_rainfall.csv')
reservoir_levels.head(5)
reservoir_rainfall.head(5)
pd.to_datetime('2018-01-15 3:45pm')
pd.to_datetime('7/8/2018')
pd.to_datetime('7/8/2018', dayfirst=True)
pd.to_datetime(['2018-01-05', '7/8/1952', 'Oct 10, 2018'])
pd.to_datetime(['2/25/10','8/6/17', '12/15/12'], format='%m/%d/%y')
reservoir_levels.dtypes
reservoir_levels['Date'] = pd.to_datetime(reservoir_levels['Date'])
reservoir_levels = reservoir_levels.set_index('Date')
reservoir_levels.head()
reservoir_levels['Year'] = reservoir_levels.index.year

reservoir_levels['Month'] = reservoir_levels.index.month

reservoir_levels['Weekday_name'] = reservoir_levels.index.weekday_name
reservoir_levels.sample(5, random_state=10)
reservoir_levels.loc['2015-05-16']
reservoir_levels['2015-05-16']
reservoir_levels.loc['2015-05-16':'2015-06-16']
reservoir_levels.loc['2015-05']
import matplotlib.pyplot as plt

import seaborn as sns



sns.set(rc={'figure.figsize':(11, 4)})
reservoir_levels.loc['2015-05',['POONDI']].plot(linewidth=0.5)

reservoir_levels.loc['2015-05',['CHOLAVARAM']].plot(linewidth=0.5)

reservoir_levels.loc['2015-05',['REDHILLS']].plot(linewidth=0.5)

reservoir_levels.loc['2015-05',['CHEMBARAMBAKKAM']].plot(linewidth=0.5)

plt.show()
reservoir_levels.loc['2015-05',['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM']].plot(linewidth=0.5)


fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

for name, ax in zip(['POONDI','CHOLAVARAM','REDHILLS','CHEMBARAMBAKKAM'], axes):

    sns.boxplot(data=reservoir_levels['2015'], x='Month', y=name, ax=ax)

    ax.set_ylabel('mcft')

    ax.set_title(name)

# Remove the automatic x-axis label from all but the bottom subplot

    if ax != axes[-1]:

        ax.set_xlabel('')
sns.boxplot(data=reservoir_levels['2015-05'], x='Weekday_name', y='REDHILLS');
pd.date_range('2019-03-10', '2019-03-15', freq='D')
pd.date_range('2019-09-20', periods=8, freq='H')
times_sample = pd.to_datetime(['2015-06-02', '2015-06-03', '2015-06-07'])



sample_df = reservoir_levels.loc[times_sample, ['POONDI']].copy()

sample_df
weekly_mean = reservoir_levels.loc['2015',['POONDI','REDHILLS']].resample('W').mean()

weekly_mean.head(3)