# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing mining tools

import seaborn as sns

import matplotlib.pyplot as plt

import pylab 

import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')
# Reading the input files

test_data = pd.read_csv("../input/test.csv")

train_data = pd.read_csv("../input/train.csv")
#reading the data description file

with open('../input/data_description.txt', 'r') as f2:

    data = f2.read()

    print(data[0:1000])
# mining the GOLDs from training data



print("Training data size:",train_data.size)

print("Total Columns: ",train_data.columns.size)

print("Target:","SalePrice")

print("Columns and data types:\n",train_data.dtypes[0:20])
#Distribution of the SalePrice data

x = train_data['SalePrice']

y = np.linspace(0, 10, len(x))

plt.figure(figsize=(16,10))

plt.xlabel("SalePrice",labelpad=10)

plt.plot(x, y, '+')
#Distribution of saleprice

plt.figure(figsize=(12,8))

sns.kdeplot(train_data.SalePrice,shade=True)
#plotted aggainst Unique SalePrice and It's Count

rng = np.random.RandomState(0)

x = train_data['SalePrice'].value_counts().to_frame().index #unique SalePrice

y = train_data['SalePrice'].value_counts().to_frame()["SalePrice"] #counts of the each unique Sales Price

colors = rng.rand(len(x))

sizes = 1000 * rng.rand(len(x))

plt.figure(figsize=(18,10))

plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,cmap='viridis')

plt.colorbar();  # show color scale
#probability plot of the SalePrice data

#Per Quantile data distribution

plt.figure(figsize=(20,6))

stats.probplot(train_data["SalePrice"], dist="norm", plot=plt)

plt.show()
#Looking at SalePrice data

# train_data[['SalePrice']]

print("Mean: ",train_data["SalePrice"].mean(),"\nMedian: ",train_data["SalePrice"].median(),"\nMode: ",train_data["SalePrice"].mode())

plt.figure(figsize=(20,6))

ax = sns.boxplot(x=train_data["SalePrice"])
#Outliers 

train_data[train_data["SalePrice"] > 600000]
#correlation between feature

corr_data = train_data.corr()

corr_data.head()
#Dependency of SalePrice on remaining features/columns

sales_corr = corr_data.iloc[-1].to_frame()[0:-1].sort_values(by='SalePrice',ascending=False)

sales_corr.head()