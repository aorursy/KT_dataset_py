import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_cov = pd.read_csv('../input/dsc-summer-school-data-visualization-challenge/2019_nCoV_20200121_20200206.csv', infer_datetime_format=True, parse_dates=['Last Update'])

df_cov.head()
df_cov.describe()
sns.set_style('dark')
sns.relplot(x='Death', y='Confirmed', data=df_cov)
df_cov['Country/Region'].unique()
sns.relplot(x='Confirmed', y='Suspected', data=df_cov, hue='Country/Region'); # hint: semicolon
sns.set_style('whitegrid')
df_cov['Province/State'].unique()
sns.relplot(x='Last Update', y='Death', data=df_cov, hue='Country/Region', kind='line', height=10);
sns.relplot(x='Confirmed', y='Death', data=df_cov, hue='Country/Region', kind='line', height=10);
sns.relplot(x='Death', y='Recovered', data=df_cov, hue='Country/Region'); # hint: semicolon
sns.relplot(x='Last Update', y='Death', data=df_cov, kind='line', height=10)
sns.relplot(x='Last Update', y='Recovered', data=df_cov, kind='line', height=10)
data = df_cov.groupby('Country/Region').mean
data
B = data.dropna()
B
sns.relplot(x='Suspected', y='Confirmed', data=B, kind='line', height=10)
z = df_cov.groupby('Country/Region')
z
df_cov.shape