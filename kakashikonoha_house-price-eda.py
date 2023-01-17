! pip install pyod
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm, skew

import matplotlib.pyplot as plt

import seaborn as sns
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import data from source

import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# Dataset: TRAIN  | Operation: Identify null values

missing_vals = pd.isnull(train).sum()

missing_vals = missing_vals[missing_vals > 0]

plt.xticks(rotation=90)

sns.barplot(x=missing_vals.index, y=missing_vals)

plt.show()
# Dataset: TEST  | Operation: Identify null values

missing_vals = pd.isnull(test).sum()

missing_vals = missing_vals[missing_vals > 0]

plt.xticks(rotation=90)

sns.barplot(x=missing_vals.index, y=missing_vals)

plt.show()
# Dataset: TEST  | Operation: matrix to find the pattern of missingness in the dataset

import missingno as msno

# msno.matrix(train.sample(500))

msno.matrix(train)
#Dataset: Train  | Operation: Categorizing neumerical and categorical cols.

n_cols = []

c_cols = []



n_cols = [col for col in train.columns if type(train[col][0]) is not str]

c_cols = [col for col in train.columns if type(train[col][0]) is str]



# Integer columns, which are categorical

cat = ['MSSubClass', 'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'MoSold'] # Handwork !



c_cols += cat

n_cols = list(set(n_cols) - set(cat) - set(['Id', 'SalePrice']))
# Dataset: Train  | Operation: Checking skewness, before and after log transform

sale_pr = pd.DataFrame({"price":train["SalePrice"], "log-price":np.log1p(train["SalePrice"])})

sale_pr.hist()

tmp = np.log1p(train["SalePrice"])
# Dataset: Train  | Operation: Mean and SD, before and after log transform

print("Before log transform:  Mean: %f, Standard Deviation: %f" %norm.fit(train['SalePrice']))

print("After log transform: Mean: %f, Standard Deviation: %f" %norm.fit(tmp))
# Dataset: Train  | Operation: features correlation with the target variable

storage = []

for col in n_cols:

    na_idx = pd.isnull(train[col])

    correlation = np.corrcoef(x= train[col][~na_idx], y=train['SalePrice'][~na_idx])[0,1]

    storage.append((col, correlation))

storage.sort(key=lambda x : -abs(x[1]))



storage[:4]
# Dataset: Train  | Operation: Graph plot of feature and SalesPrice, for top 15 correlated features.

b = [p for (p,q) in storage[0:15]]

N=15



fig, ax = plt.subplots(int(np.ceil(N/2)),2, figsize=(15,14*2))

for i, col in enumerate(b):

    sns.scatterplot(data=train, 

             x=col, 

             y="SalePrice", 

             alpha=0.4, 

             ax=ax[i//2][i%2])

    ax[i//2][i%2].set_xlabel(col, fontsize=18)

    ax[i//2][i%2].set_ylabel('SalePrice', fontsize=18)

plt.show()
# Visuals for some correlated features.



sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show()
# Dataset: Train  | Operation: correlation matrix for neumerical columns.

corr_matrix = train[n_cols].corr()

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr_matrix, vmax=.8, square=True)
# Dataset: Train  | Operation: Correlated pairs where abs(corr) > 0.4 , for neumerical columns.

top_correlations = []

tmp = corr_matrix[abs(corr_matrix)>0.4]

for col in tmp.columns:

    for row in tmp[col][~pd.isnull(tmp[col])].index:

        if col == row:

            break

        top_correlations.append((col,row, tmp[col][row]))

top_correlations.sort(key = lambda x : -x[2])

top_correlations[:5]
# Dataset: Train  | Operation: Outliers from GrLivArea

sns.scatterplot(data=train, x='GrLivArea', y='SalePrice') 

plt.show()
# Dataset: Train  | Operation: outlier detection in scaled and transformed target column

scaler = StandardScaler()

sc_tr = scaler.fit_transform(train[['SalePrice']])

sc_tr_srt = sorted(np.squeeze(sc_tr))

for a, b in zip(sc_tr_srt[:10], sc_tr_srt[-10:]):

    print('{} {} {}' .format(round(a, 5), ' '*10, round(b,5)))
# Dataset: Train  | Operation: skewness of columns

skews = []

for col in n_cols:

    skews.append((col, skew(train[col])))

skews.sort(key=lambda x : -abs(x[1]))
# Dataset: Train  | Operation: plot before log transform

fig, ax = plt.subplots(1,2, figsize=(15,5))

sns.scatterplot(data=train, x='GrLivArea', y='SalePrice', ax=ax[0])

sns.scatterplot(data=train, x='GarageArea', y='SalePrice', ax=ax[1])

plt.show()
# Dataset: Train  | Operation: plot after log transform

fig, ax = plt.subplots(1,2, figsize=(15,5))

dat = train.copy()

dat['SalePrice'] = np.log1p(dat['SalePrice'])

dat['GrLivArea'] = np.log1p(dat['GrLivArea'])

dat['MasVnrArea'] = np.log1p(dat['MasVnrArea'])

sns.scatterplot(data=dat, x='GrLivArea', y='SalePrice', ax=ax[0])

sns.scatterplot(data=dat, x='GarageArea', y='SalePrice', ax=ax[1])

plt.show()