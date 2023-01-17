# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



#Setting Style for Plotting

plt.style.use('fivethirtyeight')
df = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
df.shape
df.info()
df.describe()
df.quality.unique()
df.quality.value_counts()
df['quality'].hist()
fig, ax = plt.subplots(figsize=(15,7))

sns.heatmap(df.corr(),cmap='viridis', annot=True)
l = df.columns.values

number_of_columns=12

number_of_rows = len(l)-1/number_of_columns

plt.figure(figsize=(number_of_columns,5*number_of_rows))

for i in range(0,len(l)):

    plt.subplot(number_of_rows + 1,number_of_columns,i+1)

    sns.set_style('whitegrid')

    sns.boxplot(df[l[i]],color='green',orient='v')

    plt.tight_layout()
plt.figure(figsize=(2*number_of_columns,5*number_of_rows))

for i in range(0,len(l)):

    plt.subplot(number_of_rows + 1,number_of_columns,i+1)

    sns.distplot(df[l[i]],kde=True) 
print("Skewness  \n ",df.skew())

print("\n Kurtosis  \n ", df.kurt())
# Read and load Data

train = pd.read_csv("../input/housepricesadvancedregressiontechniquestrain/train.csv")

train.describe()
#Plot Histogram for 'SalePrice'

sns.distplot(train['SalePrice'])
# Skewness and Kurtosis

print("Skewness : %f" % train['SalePrice'].skew())

print("Kurtosis : %f" % train['SalePrice'].kurt())
target = np.log(train.SalePrice)

print("Skewness : %f" % target.skew())

print("Kurtosis : %f" % target.kurt())
train = train[train['GarageArea'] < 1200]

# Histogram and normal probability plot

import seaborn as sns

from scipy import stats

from scipy.stats import norm



sns.distplot(train['SalePrice'], fit = norm)

fig = plt.figure()

res = stats.probplot(train['SalePrice'],plot = plt)