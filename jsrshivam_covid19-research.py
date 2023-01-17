import torch

import torch.nn as nn



import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
train_data_raw = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
train_data.head()
train_data[train_data['Country_Region'] == 'India']
train_data.dtypes
train_data['Date']= pd.to_datetime(train_data['Date']) 
train_data.info()
train_data.head(1)
train_data = train_data.set_index("Date")
train_data[train_data['Country_Region'] == 'India'].head()
train_data['Year'] = train_data.index.year

train_data['Month'] = train_data.index.month

train_data['Weekday Name'] = train_data.index.weekday_name

# Display a random sampling of 5 rows

train_data.sample(5, random_state=0)
train_data.loc['2020-04-01']
sns.set(rc={'figure.figsize':(20, 10)})
train_data[train_data['Country_Region'] == 'India'][['ConfirmedCases', 'Fatalities']].plot(linewidth=2);




cols_plot = ['ConfirmedCases', 'Fatalities', ]

axes = train_data[cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

for ax in axes:

    ax.set_ylabel('Daily Totals')


cols_plot = ['ConfirmedCases', 'Fatalities', ]

axes = train_data[train_data['Country_Region']=='India'][cols_plot].plot(marker='.', alpha=0.5, linestyle='None', figsize=(11, 9), subplots=True)

for ax in axes:

    ax.set_ylabel('Daily Totals')

    ax.set_ylim(0,20000)
sns.boxplot(data=train_data[train_data['Country_Region'] == 'India'], x='Weekday Name', y='ConfirmedCases');
sns.boxplot(data=train_data[train_data['Country_Region'] == 'India'], x='Weekday Name', y='Fatalities');
sns.boxplot(data=train_data[train_data['Country_Region'] == 'Italy'], x='Weekday Name', y='ConfirmedCases');
sns.boxplot(data=train_data[train_data['Country_Region'] == 'Uruguay'], x='Weekday Name', y='ConfirmedCases');
sns.boxplot(data=train_data[train_data['Country_Region'] == 'Uruguay'], x='Weekday Name', y='Fatalities');
sns.boxplot(data=train_data[train_data['Country_Region'] == 'Spain'], x='Weekday Name', y='ConfirmedCases');
sns.boxplot(data=train_data[train_data['Country_Region'] == 'Spain'], x='Weekday Name', y='Fatalities');