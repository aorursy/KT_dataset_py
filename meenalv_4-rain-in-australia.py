import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
weather_df = pd.read_csv("/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv")

print(weather_df.info())

weather_df.head()
def getUnique(df):

    for col in df.columns:

        print(col + " : " + str(df[col].nunique()) + " ; " + str(df[col].count()) + " ; " + str(df[col].isnull().sum()))
getUnique(weather_df)
df = weather_df['MinTemp'].value_counts()

df
fig_dims = (14, 5)

sns.set()

fig, axes = plt.subplots( figsize=fig_dims)

sns.barplot(df[:10,].index, df[:10,].values)
df_max = weather_df['MaxTemp'].value_counts()

df_max
fig_dims = (14, 5)

sns.set()

fig, axes = plt.subplots( figsize=fig_dims)

sns.barplot(df_max[:10, ].index, df_max[:10,].values)
df_rain = weather_df[weather_df['Rainfall'] != 0.0]['Rainfall'].value_counts()

df_rain
fig_dims = (14, 5)

sns.set()

fig, axes = plt.subplots( figsize=fig_dims)

sns.barplot(df_rain[:15, ].index, df_rain[:15,].values)
weather_df[weather_df['Rainfall'] == 0.2].head()
df_rainToday = weather_df[weather_df['RainToday'] == 'Yes']['Rainfall'].value_counts()

df_rainToday
fig_dims = (14, 5)

sns.set()

fig, axes = plt.subplots( figsize=fig_dims)

sns.barplot(df_rainToday[:15, ].index, df_rainToday[:15,].values)
weather_df.sort_values('Rainfall' , ascending = False)