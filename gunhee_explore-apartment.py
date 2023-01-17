import pandas as pd

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

df = pd.read_csv('../input/Daegu_Real_Estate_data.csv')

df.head()
df.isnull().sum()
df.info()
df.describe()
# function to divide data by data type



def div_cols(df):

    df_division = {'number': [], 'string': []}

    for i in df.columns:

        # Numeric data

        if df[i].dtype=='int64':

            df_division['number'].append(i)

        elif df[i].dtype=='float64':

            df_division['number'].append(i)

        # Categorical data

        else:

            df_division['string'].append(i)

    return df_division
# divide data

df_division = div_cols(df)
# plot numeric data



for i in df_division['number']:

    plt.hist(df[i])

    plt.title(i)

    plt.show()
# plot categorical data



for i in df_division['string']:

    # count frequency

    dic = Counter(df[i])

    k = dic.keys()

    v = dic.values()

    

    # set plt options

    y_pos = np.arange(len(k))

    plt.bar(y_pos, v, align='center')

    plt.xticks(y_pos, k)

    plt.ylabel('Frequency')

    plt.xlabel(i)

    plt.title(i)

    plt.show()
plt.subplots(figsize=(10,6))

sns.boxplot(x='HallwayType', y='YearBuilt', data=df)
sns.boxplot(x='HeatingType', y='YearBuilt', data=df)
sns.boxplot(x='AptManageType', y='YearBuilt', data=df)
sns.boxplot(x='TimeToBusStop', y='SalePrice', data=df)
plt.subplots(figsize=(15,10))

sns.boxplot(x='TimeToSubway', y='SalePrice', data=df)
from statsmodels.graphics import mosaicplot

ax = mosaicplot.mosaic(df, ['TimeToSubway', 'TimeToBusStop'])
# create the matplotlib Figure and Axes objects ahead of time to set fig size

fig, ax = plt.subplots(figsize=(15, 10))

sns.boxplot(x='SubwayStation', y='SalePrice', data=df, ax=ax)
# correlation between Price

df.corr()['SalePrice']
# correlation heatmap



plt.subplots(figsize=(20,15))

corr = df.corr()

sns.heatmap(corr, cmap="YlGnBu")
sns.pointplot(x='N_FacilitiesInApt', y='SalePrice', data=df)
plt.subplots(figsize=(17,15))

sns.pointplot(x='YearBuilt', y='N_FacilitiesInApt', data=df)

df.corr()['N_FacilitiesInApt']
df.corr()['N_FacilitiesNearBy(PublicOffice)']
# redline: number of parkinglot on the ground

# blueline: number of parkinglot under the ground



plt.subplots(figsize=(17,15))

sns.pointplot(x='YearBuilt', y='N_Parkinglot(Ground)', data=df, color="red")

sns.pointplot(x='YearBuilt', y='N_Parkinglot(Basement)', data=df)
fig, ax = plt.subplots(figsize=(18, 10))

sns.stripplot(x='Floor', y='SalePrice', data=df)
plt.subplots(figsize=(15,10))

sns.boxplot(x='YrSold', y='SalePrice', data=df)