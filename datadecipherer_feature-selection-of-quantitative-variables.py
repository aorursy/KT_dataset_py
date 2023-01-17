

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy.stats import norm



from scipy import stats

import warnings



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

tr = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

pd.set_option('display.max_columns', None)

tr.head()

tr.info()
tr.groupby('LotShape')
tr

tr.groupby('MSZoning')
tr.drop('Alley', axis = 1)
tr1 = tr.drop(['Alley','LotFrontage','FireplaceQu','PoolQC','Fence','MiscFeature'], axis = 1)


pd.value_counts(tr['LotShape'])
tr1['SalePrice'].describe()
sns.distplot(tr1['SalePrice']);
d1 = pd.concat([tr['SalePrice'], tr['GrLivArea']], axis=1)

d1.plot.scatter(x="GrLivArea", y='SalePrice', ylim=(0,800000));\

d1
var = 'TotalBsmtSF'

data = pd.concat([tr1['SalePrice'], tr1[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
corr = tr1.corr()  #creating the correlation matrix

f,ax = plt.subplots(figsize=(12, 9))  # Assigning the size of the plot

hp = sns.heatmap(corr, cmap = 'YlGnBu')  #Creating the Heatmap og the correlation matrix




#saleprice correlation matrix #Creating a corr matrix using 10 columns that have the highest correlation with SalePrice

k = 10 #number of variables for heatmap

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(tr1[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, cmap = 'YlGnBu')

plt.show()

cols



#handling missing data



totalmissingcount = tr1.isnull().sum().sort_values(ascending=False)

percent = (tr1.isnull().sum()/tr.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([totalmissingcount, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
df_train = tr1

df_train.shape
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train
# DEALING WITH MISSING VALUES

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index) #removing only one null value from electrical

df_train.isnull().sum().max()
tr3 = df_train[cols]

tr3
last = tr3['SalePrice']

tr3.drop(labels=['SalePrice'], axis=1, inplace = True)

tr3.insert(9, 'SalePrice', last)

tr3
y = tr3[['SalePrice']]

y

x = tr3[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt']]

x