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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.drop(columns = 'Id',axis = 1,inplace = True)

#correlation Matrix

corrmat = train.corr()



f,ax = plt.subplots(figsize = (20,9))

sns.heatmap(corrmat, vmax = .8)
corrmat.columns
#here we can se how the feature are correlated

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols],size = 2.2)

plt.show()
from scipy import stats

from scipy.stats import skew, norm



sns.distplot(train['SalePrice'],fit = norm)



#Get the fitted parameter used by the function



(mu, sigma) = norm.fit(train['SalePrice'])

print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu,sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)

plt.show()
train['SalePrice'] = np.log(train['SalePrice'])
from scipy.stats import skew

skewness = train.apply(lambda x : skew(x))

skewness.sort_values(ascending = False)
#data has been normalized

sns.distplot(train['SalePrice'],fit = norm)
#Differentiate between Categorial and Numerical Variable



categorical_feature = train.select_dtypes(include = ['object']).columns

numerical_features = train.select_dtypes(exclude = ['object']).columns
categorical_data = train.select_dtypes(include = ['object'])

numerical_data  = train.select_dtypes(exclude = ['object'])
from scipy.stats import skew

skewness = numerical_data.apply(lambda x : skew(x))

skewness.sort_values(ascending = False)
f,ax = plt.subplots(figsize = (20,9))



sns.heatmap(numerical_data.corr(),vmax = 0.8,annot = True)