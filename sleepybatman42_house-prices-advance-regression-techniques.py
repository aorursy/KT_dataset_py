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
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from scipy import stats

from scipy.stats import norm



from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
test.head()
train.shape

#(1460, 81)
train.describe()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False)
train['SalePrice'].describe()
#Removing ID as it is not neccessary

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)
train.shape #with ID removed

test.shape 

#Train 1460, 80

#Test 1459, 79
#Missing Values

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
#Removing Missing Values

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train = train.drop(train.loc[train['Electrical'].isnull()].index)

train.isnull().sum().max()
saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
sns.distplot(train['SalePrice']);
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=15)

plt.xlabel('GrLivArea', fontsize=15)

plt.show()
sns.distplot(train['SalePrice'], fit=norm)

plt.ylabel('Frequency')

plt.title('SalePrice distribution');

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)
#utilizing log1p to correct skew in previous probability plot

train["SalePrice"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePrice'], fit=norm)

plt.ylabel('Frequency')

plt.title('SalePrice distribution');

fig = plt.figure()

res = stats.probplot(train['SalePrice'], plot=plt)


sns.distplot(train['GrLivArea'], fit=norm)

plt.ylabel('Frequency')

plt.title('GrLivArea distribution');

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
#utilizing log1p to correct skew in previous probability plot

train["GrLivArea"] = np.log1p(train["GrLivArea"])
sns.distplot(train['GrLivArea'], fit=norm)

plt.ylabel('Frequency')

plt.title('GrLivArea distribution');

fig = plt.figure()

res = stats.probplot(train['GrLivArea'], plot=plt)
sns.distplot(train['TotalBsmtSF'], fit=norm)

plt.ylabel('Frequency')

plt.title('TotalBsmtSF distribution');

fig = plt.figure()

res = stats.probplot(train['TotalBsmtSF'], plot=plt)
#To correct this skew log1 will no longer work, and I will need to make a new variable

train['HasBsmt'] = pd.Series(len(train['TotalBsmtSF']), index=train.index)

train['HasBsmt'] = 0 

train.loc[train['TotalBsmtSF']>0,'HasBsmt'] = 1



train.loc[train['HasBsmt']==1,'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])
sns.distplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)

plt.ylabel('Frequency')

plt.title('TotalBsmtSF distribution');

fig = plt.figure()

res = stats.probplot(train[train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
train = pd.get_dummies(train)

print(train.shape)
test.head()
all_data = pd.concat([train,test],axis=0)

y_train = train.SalePrice.values
all_data.shape
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)



score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))