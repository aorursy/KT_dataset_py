import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

print(os.listdir("../input"))
df = pd.read_csv('../input/metro-bike-share-trip-data.csv')

df.columns
df.head()

#Most of the columns after 'Passholder Type' appear null. Let's check the end part of data.
df.tail()

#There are infact some values in these columns which appear null in first peek.
df.isnull().sum()
df['Plan Duration'].unique()

#Let's check corresponding which PassholderType is the value nan.
df[df['Plan Duration'].isnull()]['Passholder Type']

#Here we can easily fill the nan values in 'Plan Duration' using the Passholder Type

#Staff Annual: 365 days

#Monthly Pass: 30 days

for index in df[df['Plan Duration'].isnull()].index:

    if df.loc[index, 'Passholder Type']=='Staff Annual':

        df.loc[index, 'Plan Duration']=365

    elif df.loc[index, 'Passholder Type']=='Monthly Pass':

        df.loc[index, 'Plan Duration']=30
df[df['Plan Duration'].isnull()]

#The null values have been filled up.
df['Starting Station Latitude'].index == df['Starting Station Longitude'].index

df['Ending Station Latitude'].index == df['Ending Station Longitude'].index

#values of latitude and longitude are missing in the same rows.
#I will drop the columns which I don't need for now.

df = df.drop(['Starting Lat-Long', 'Ending Lat-Long', 'Neighborhood Councils (Certified)', 

              'Council Districts', 'Zip Codes', 'LA Specific Plans', 'Precinct Boundaries', 

              'Census Tracts', 'Bike ID', 'Plan Duration', 'Trip ID'], axis=1)
#Given the number of rows is big we can afford to remove those rows which contain null values.

df = df.dropna()
df.isnull().sum()
def count_plot(value):

    sns.countplot(x=value, data=df)



count_plot('Trip Route Category')
#Monthly pass is the most popular choice

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

count_plot('Passholder Type')

plt.subplot(1, 2, 2)

sns.countplot(x='Passholder Type', data=df, hue='Trip Route Category')
df['Start Time'] = pd.to_datetime(df['Start Time'])

df['End Time'] = pd.to_datetime(df['End Time'])
sdf = df.groupby('Starting Station ID').agg('max')

len(df['Trip ID'].unique())