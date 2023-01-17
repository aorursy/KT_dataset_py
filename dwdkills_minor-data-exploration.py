# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

sns.set(style='ticks')
data = pd.read_csv("../input/weather/weather.csv", decimal=',')
data.head()
def detailed_analysis(df): 

    obs = df.shape[0]

    types = df.dtypes

    counts = df.apply(lambda x: x.count())

    uniques = df.apply(lambda x: [x.unique()])

    nulls = df.apply(lambda x: x.isnull().sum())

    distincts = df.apply(lambda x: x.unique().shape[0])

    missing_ratio = (df.isnull().sum() / obs) * 100

    skewness = df.skew()

    kurtosis = df.kurt() 

    print('Data shape:', df.shape)

    

    cols = ['types', 'counts', 'distincts', 'nulls', 'missing ratio', 'uniques', 'skewness', 'kurtosis']

    details = pd.concat([types, counts, distincts, nulls, missing_ratio, uniques, skewness, kurtosis], axis = 1)

    

    details.columns = cols

    dtypes = details.types.value_counts()

    print('___________________________\nData types:\n',dtypes)

    print('___________________________')

    return details
details = detailed_analysis(data)

details
correlations = data.corr()



fig = plt.figure(figsize=(12, 10))

sns.heatmap(correlations, annot=True, cmap='YlOrRd')
fig = plt.figure(figsize=(15, 10))

sns.regplot(x='temperature', y='humidity', data=data)
fig = plt.figure(figsize=(15, 10))

sns.boxplot(x='weather', y='temperature', hue='fire', data=data)
fig = plt.figure(figsize=(15, 10))

sns.barplot(x='weather', y='visibility', data=data, ci=None)
data[['weather', 'temperature']].groupby('weather').mean().sort_values(by='temperature', ascending=False)
data[['fire', 'wind']].groupby('fire').mean().sort_values(by='wind', ascending=False)