# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt #plottting

import seaborn as sns # more plotting



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Load in only the training data as this is what we will be exploring 

train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
# Print the features and the first 5 rows of the dataset

print(train_data.columns.values)

train_data.head(5)
# Print the length of the data

print("# of training Rows = ", len(train_data))



# Check for NaNs in the data

print("NaNs in each training Feature")

dfNull = train_data.isnull().sum().to_frame('nulls')

print(dfNull.loc[dfNull['nulls'] > 0]) # Print only features that have Null values
# Drop the features mentioned above

train_data.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True)



# Checking if any features take all the same value in which case they could be removed (in this case none are removed)

for i in train_data.keys():

    if len(pd.unique(train_data[i])) < 2:

        print(i)
# Plot a heatmap of the correlations for a quick check on the numerical features

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(train_data.corr(), annot = False, cmap='viridis', square=True)
# Make histogram of SalePrice using seaborn

sns.set(color_codes=True)

fig, ax = plt.subplots(1,1, figsize=(15,6))

sns.distplot(train_data['SalePrice']/1000.,ax=ax, bins=40, rug=True) # Plot price in $1000s

ax.set(xlabel='SalePrice ($1000s)')
# Find the 10 largest correlations to SalePrice

bigCorr = train_data.corr().nlargest(10, 'SalePrice')['SalePrice']

print(bigCorr)



# Find the 10 largest anti-correlations to SalePrice

bigAnti = train_data.corr().nsmallest(10, 'SalePrice')['SalePrice']

print(bigAnti)
sns.pairplot(train_data[bigCorr.index])
train_data.hist(figsize=(16,20), bins=20)
# Make a list of the continuous and categorical features to plot

continuous_keys = ['SalePrice', 'LotArea']

categorical_keys = ['OverallCond', 'OverallQual']



# Iterate through each feature and plot

for i in continuous_keys:

    sns.catplot(x = 'ExterQual',y=i, data = train_data)

for i in categorical_keys:

    sns.catplot(x = 'ExterQual',y=i, kind='bar', data = train_data)
# Find the 10 largest correlations to SalePrice

bigCorr = train_data.corr().nlargest(10, 'LotFrontage')['LotFrontage']

print(bigCorr)



# Find the 10 largest anti-correlations to SalePrice

bigAnti = train_data.corr().nsmallest(10, 'LotFrontage')['LotFrontage']

print(bigAnti)