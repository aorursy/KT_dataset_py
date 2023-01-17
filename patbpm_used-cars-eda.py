# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pylab as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/craigslist-carstrucks-data/vehicles.csv')
df.shape
df.head(2)
df.columns
df.nunique(axis=0)
df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
df.condition.unique()

# Reclassify condition column



def clean_condition(row):

    

    good = ['good','fair']

    excellent = ['excellent','like new']       

    

    if row.condition in good:

        return 'good'   

    if row.condition in excellent:

        return 'excellent'    

    return row.condition# Clean dataframe

def clean_df(playlist):

    df_cleaned = df.copy()

    df_cleaned['condition'] = df_cleaned.apply(lambda row: clean_condition(row), axis=1)

    return df_cleaned# Get df with reclassfied 'condition' column

df_cleaned = clean_df(df)

print(df_cleaned.condition.unique())
df_cleaned.shape
df_cleaned = df_cleaned.copy().drop(['url','region_url','image_url'],axis=1)
df_cleaned.isna().sum()
df_cleaned.shape[0]
#I used the following code to remove any columns that had 40% or more of its data as null values. 

NA_val = df_cleaned.isna().sum()

def na_filter(na, threshold = .4): #only select variables that passees the threshold

    col_pass = []

    for i in na.keys():

        if na[i]/df_cleaned.shape[0]<threshold:

            col_pass.append(i)

    return col_pass

df_cleaned = df_cleaned[na_filter(NA_val)]

df_cleaned.columns
df_cleaned.shape
(df_cleaned['year'] > 1990).head()
df_cleaned = df_cleaned[df_cleaned['price'].between(999.99, 99999.00)]

df_cleaned = df_cleaned[df_cleaned['year'] > 1990]

df_cleaned = df_cleaned[df_cleaned['odometer'] < 899999.00]

df_cleaned.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
df_cleaned.shape
df_cleaned = df_cleaned.dropna(axis=0)

df_cleaned.shape
df_cleaned.corr()
#I used sns.heatmap() 

#to plot a correlation matrix of all of the variables in the used car dataset



# calculate correlation matrix

corr = df_cleaned.corr()

# plot the heatmap

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
#Scatter plot

df_cleaned.plot(kind='scatter', x='odometer', y='price')
df_cleaned.plot(kind='scatter', x='year', y='price')
#sns.pairplot() is a great way to create scatterplots 

#between all of your variables



sns.pairplot(df_cleaned)
df_cleaned['odometer'].plot(kind='hist', bins=50, figsize=(12,6), facecolor='grey',edgecolor='black')

df_cleaned['year'].plot(kind='hist', bins=20, figsize=(12,6), facecolor='grey',edgecolor='black')
df_cleaned.boxplot('price')