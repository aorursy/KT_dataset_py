# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.describe()
df.head(10)
#missing data

null_total = df.isnull().sum().sort_values(ascending=False)

null_percentage = (null_total/df.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([null_total, null_percentage], axis=1, keys=['Total', 'Percent'])

missing_data.head(10)
# Lets drop those features



df = df.drop(['PoolQC','MiscFeature','Alley','Fence'],axis = 1)
# importing some necesaary stuff

import matplotlib.pyplot as plt

import seaborn as sns



corr = df.corr()

f, ax = plt.subplots(figsize=(15, 12))

sns.heatmap(corr, vmax=.8, square=True);
k = 10 #This is for determining top 10 important features

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
df_train = df[['OverallQual','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','GarageCars','GarageArea','YearBuilt','TotRmsAbvGrd']]
print(df.shape)

df.head(10)
xtrain = df_train

ytrain = df['SalePrice']
xtrain.head(5)
ytrain.head(5)
print(xtrain.shape)

print(ytrain.shape)