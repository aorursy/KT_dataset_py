# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/photovideofixationtransportsystemmoscow/data-400-2019-12-16.csv',sep = ';')
data.info()
data = data.drop(['Unnamed: 13'],axis =1)
data.columns
data.head(10)
data['ReportDate'] = pd.to_datetime(data['ReportDate'] ,format='%d.%m.%Y', errors='coerce')



for columnname in data.columns:

    if columnname!= 'ReportDate':

        data[columnname]  = pd.to_numeric(data[columnname],downcast='integer', errors = 'coerce')

        data[columnname] = data[columnname].fillna(0)

        data[columnname] = data[columnname].astype('int')
data['month']  = data['ReportDate'].dt.month

data['weekday']  = data['ReportDate'].dt.weekday

data['year']= data['ReportDate'].dt.year

data['year'] =data['year'].fillna(0)

data['year'] = data['year'].astype('int')
data = data.loc[(data['year']<=2020)&(data['year']>2012)]
data.boxplot(['TotalInfringementsAmount',

              'SpeedLimitInfringementsAmount',

              'BusLaneOffenceAmount',

            ],figsize = [10,4])
data.boxplot(['TTKOffenceAmount',

             'ParkingOffenceAmount'],figsize = [10,4])
data.boxplot(['FormalizedTotalInfringements',

       'FormalizedSpeedLimitInfringementsAmount'],figsize = [10,4]

            )
data.boxplot(['FormalizedBusLaneOffenceAmount', 'FormalizedTTKOffenceAmount',

       'FormalizedParkingOffenceAmount'],figsize=[10,4])
data.pivot_table(index = 'year', values = ['TotalInfringementsAmount','SpeedLimitInfringementsAmount', 

                                      'BusLaneOffenceAmount','TTKOffenceAmount', 'ParkingOffenceAmount']).plot(figsize = [13,7],grid = True)
data.pivot_table(index = 'month', values = ['TotalInfringementsAmount','SpeedLimitInfringementsAmount', 

                                      'BusLaneOffenceAmount','TTKOffenceAmount', 'ParkingOffenceAmount']).plot(figsize = [13,7],grid = True)
data.pivot_table(index = 'weekday', values = ['TotalInfringementsAmount','SpeedLimitInfringementsAmount', 

                                      'BusLaneOffenceAmount','TTKOffenceAmount', 'ParkingOffenceAmount']).plot(figsize = [13,7],grid = True)
data.pivot_table(index = 'year', values = ['FormalizedTotalInfringements',

       'FormalizedSpeedLimitInfringementsAmount',

       'FormalizedBusLaneOffenceAmount', 'FormalizedTTKOffenceAmount',

       'FormalizedParkingOffenceAmount']).plot(figsize = [13,7],grid = True)
data.pivot_table(index = 'month', values = ['FormalizedTotalInfringements',

       'FormalizedSpeedLimitInfringementsAmount',

       'FormalizedBusLaneOffenceAmount', 'FormalizedTTKOffenceAmount',

       'FormalizedParkingOffenceAmount']).plot(figsize = [13,7],grid = True)
data.pivot_table(index = 'weekday', values = ['FormalizedTotalInfringements',

       'FormalizedSpeedLimitInfringementsAmount',

       'FormalizedBusLaneOffenceAmount', 'FormalizedTTKOffenceAmount',

       'FormalizedParkingOffenceAmount']).plot(figsize = [13,7],grid = True)
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

sns.despine(left=True)

sns.distplot(data['TotalInfringementsAmount'], hist=False, color="g", kde_kws={"shade": True},ax = axes[0])

sns.distplot(data['SpeedLimitInfringementsAmount'], hist=False, color="r", kde_kws={"shade": True},ax = axes[1])

sns.distplot(data['BusLaneOffenceAmount'], hist=False, color="b", kde_kws={"shade": True})

plt.setp(axes, yticks=[])

plt.tight_layout()
sns.set(style="white", palette="muted", color_codes=True)

f, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

sns.despine(left=True)

sns.distplot(data['TTKOffenceAmount'], hist=False, color="g", kde_kws={"shade": True},ax = axes[0])

sns.distplot(data['ParkingOffenceAmount'], hist=False, color="r", kde_kws={"shade": True},ax = axes[1])

plt.setp(axes, yticks=[])

plt.tight_layout()
data.head(30)