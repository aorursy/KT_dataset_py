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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
import pandas as pd

housetrain = pd.read_csv("../input/house-prices-advanced-regression-techniques/housetrain.csv")
housetrain.head()
housetrain.shape
housetrain.isnull().any()
housetrain.select_dtypes(['object'])
housetrain.columns
housetrain.info()
housetrain.describe()
sns.pairplot(housetrain)
plt.figure(figsize=(21,20))

sns.heatmap(housetrain.corr(),annot=True)
housetrain['SalePrice'].describe()
housetrain['SalePrice'].value_counts()
sns.distplot(housetrain['SalePrice'])
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([housetrain['SalePrice'], housetrain[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var='TotalBsmtSF'

data=pd.concat([housetrain['SalePrice'],housetrain[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,8000000))
#box plot overallqual/saleprice

var='OverallQual'

data=pd.concat([housetrain['SalePrice'],housetrain[var]],axis=1)

f,ax=plt.subplots(figsize=(8,6))

fig=sns.boxplot(x=var,y='SalePrice',data=data)

fig.axis(ymin=0,ymax=800000)
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(housetrain[cols], size = 2.5)

plt.show();
total=housetrain.isnull().sum().sort_values(ascending=False)

percent=(housetrain.isnull().sum()/housetrain.isnull().count()).sort_values(ascending=False)

missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])

missing_data.head(20)
saleprice_scaled=StandardScaler().fit_transform(housetrain['SalePrice'][:,np.newaxis])

low_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range=saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#bivariate analysis saleprice/grlivarea

var='GrLivArea'

data=pd.concat([housetrain['SalePrice'],housetrain[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,80000))
#Deleting points

housetrain.sort_values(by='GrLivArea',ascending=False)[:2]

housetrain=housetrain.drop(housetrain[housetrain['Id']==1299].index)

housetrain=housetrain.drop(housetrain[housetrain['Id']==524].index)
#bivariate analysis saleprice/grlivarea

var = 'TotalBsmtSF'

data = pd.concat([housetrain['SalePrice'], housetrain[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#histogram and normal probability plot

sns.distplot(housetrain['SalePrice'],fit=norm)

fig=plt.figure()

res=stats.probplot(housetrain['SalePrice'],plot=plt)
#applying log transformation

housetrain['SalePrice'] = np.log(housetrain['SalePrice'])
#transformed histogram and normal probability plot

sns.distplot(housetrain['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(housetrain['SalePrice'], plot=plt)
#histogram and normal probability plot

sns.distplot(housetrain['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(housetrain['GrLivArea'], plot=plt)