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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
data_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/housetrain.csv")
data_train.head(1299)
data_train.columns
data_train['SalePrice'].describe()
sns.distplot(data_train['SalePrice']);
print('Skewness: %f' % data_train['SalePrice'].skew())

print("Kurtosis: %f" % data_train['SalePrice'].kurt())
var = 'GrLivArea'

data = pd.concat([data_train['SalePrice'], data_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice',ylim=(0.800000));
var = 'TotalBsmtSF'

data = pd.concat([data_train['SalePrice'], data_train[var]], axis = 1)

data.plot.scatter(x=var, y='SalePrice',ylim=(0.800000));
var = 'OverallQual'

data = pd.concat([data_train['SalePrice'], data_train[var]],axis=1)

f,ax =plt.subplots(figsize=(8,6))

fig =sns.boxplot(x=var,y='SalePrice',data=data)

fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'

data = pd.concat([data_train['SalePrice'],data_train[var]],axis = 1)

f, ax = plt.subplots(figsize =(16,8))

fig = sns.boxplot(x=var,y='SalePrice',data=data)

fig.axis(ymin=0,ymax=800000)

plt.xticks(rotation=90);
#correlation matrix

corrmat = data_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
k = 10

cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index

cm = np.corrcoef(data_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)

plt.show()
total = data_train.isnull().sum().sort_values(ascending=False)

percent = (data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)

missing_data= pd.concat([total,percent],axis =1,keys=['Total','Percent'])

missing_data.head(20)
data_train = data_train.drop((missing_data[missing_data['Total'] >1]).index,1)

data_train = data_train.drop(data_train.loc[data_train['Electrical'].isnull()].index)

data_train.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(data_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution')

print(low_range)

print('\nouter raneg (high) of the distribution:')

print(high_range)
var = 'GrLivArea'

data = pd.concat([data_train["SalePrice"],data_train[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim = (0.800000));
data_train.sort_values(by = 'GrLivArea',ascending = False)[:2]

data_train = data_train.drop(data_train[data_train['Id']== 1299].index)

data_train = data_train.drop(data_train[data_train['Id']== 524].index)
data_train.head(526)
var = 'TotalBsmtSF'

data = pd.concat([data_train['SalePrice'],data_train[var]],axis = 1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000));
sns.distplot(data_train["SalePrice"],fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['SalePrice'],plot=plt)
data_train['SalePrice'] =np.log(data_train['SalePrice'])
sns.distplot(data_train['SalePrice'],fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['SalePrice'],plot=plt)
sns.distplot(data_train['GrLivArea'],fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['GrLivArea'],plot=plt)
data_train['GrLivArea']=np.log(data_train['GrLivArea'])
sns.distplot(data_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train['GrLivArea'], plot=plt)
data_train['HasBsmt'] = pd.Series(len(data_train['TotalBsmtSF']),index=data_train.index)

data_train['HasBsmt'] =0

data_train.loc[data_train['TotalBsmtSF']>0,'HasBsmt'] =1
data_train.loc[data_train['HasBsmt']==1,'TotalBsmtSF']=np.log(data_train['TotalBsmtSF'])


sns.distplot(data_train[data_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

fig = plt.figure()

res = stats.probplot(data_train[data_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
data_train = pd.get_dummies(data_train)