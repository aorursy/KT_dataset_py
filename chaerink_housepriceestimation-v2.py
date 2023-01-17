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

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
import matplotlib.pyplot as plt

import missingno as msno

from scipy import stats

from scipy.stats import norm, skew

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
train_label = train[['Id', 'SalePrice']]

features = pd.concat([train.drop(columns=['SalePrice']), test])

features.shape, train_label.shape, train.shape
msno.matrix(features)
features['totalArea'] = features['TotalBsmtSF'] + features['GrLivArea']



## 리모델링 이후에 판매되었으면 건설시점 기준으로 바꿔줘야함!!

temp = []

for ind, val in features.iterrows():

    if val['YearRemodAdd'] > val['YrSold']:

        temp.append(val['YrSold']-val['YearBuilt'])

    else:

        temp.append(val['YrSold']-val['YearRemodAdd'])





features['SinceRemod'] = temp









c_features = features[['Id', 'totalArea', 'MSSubClass', 'Neighborhood', 'SaleType', 'Heating', 'SinceRemod', 'OverallQual', 'ExterQual', 'TotRmsAbvGrd', 'GarageCars']]

c_features.head()
c_features['SinceRemod'].describe()
c_features[c_features['SinceRemod']<0]
features.iloc[2549][['YearBuilt', 'YearRemodAdd', 'YrSold']]
train_label.tail()
x = train_label['SalePrice']



plt.figure(figsize=(16,9))

plt.title("SalePrice Dist")

sns.distplot(x)

plt.show()
X = features.iloc[:1460]['totalArea']

y = train_label['SalePrice']



plt.figure(figsize=(16,9))

plt.scatter(X, y, s=5, color='dodgerblue')

plt.title('Area-Price')

plt.show()
X = features.iloc[:1460]['totalArea']

y = np.log(train_label['SalePrice'])



plt.figure(figsize=(16,9))

plt.scatter(X, y, s=5, color='dodgerblue')

plt.title('Area-Price')

plt.axvline(7500, ls='--', color='#070707')

plt.show()
features[features['totalArea']>7500]
c_features['SinceRemod'] = c_features['SinceRemod'].replace(-1, 0)
c_features.sample(5)
train_label.sample(5)
train_label['SalePrice_log'] = np.log1p(train_label['SalePrice'])



plt.figure(figsize=(9,6))

stats.probplot(train_label['SalePrice'], plot=plt)

plt.show()



plt.figure(figsize=(9,6))

stats.probplot(train_label['SalePrice_log'], plot=plt)

plt.show()

c_features.reset_index(inplace=True, drop=True)

c_features = c_features.drop(index=[523, 1298])
c_features.shape
train.shape[0], test.shape[0], train.shape[0] +test.shape[0]
train_label.drop(index=[523, 1298], inplace=True)

train_label.shape
c_features.isna().sum()
c_features[np.isnan(c_features['totalArea'])]
c_features[np.isnan(c_features['GarageCars'])]
c_features.SaleType.sort_values()
test[test['Id']==2121][['GrLivArea', 'TotalBsmtSF']]
test.columns
test[test['Id']==2121][['GrLivArea', 'TotalBsmtSF', 'BsmtQual', 'BsmtCond', 'BsmtExposure']]
temp = c_features['totalArea'].replace(np.nan, 896)

c_features['totalArea'] = temp
test[test['Id']==2577][['GarageType', 'GarageYrBlt', 'GarageArea', 'GarageQual', 'GarageCond']]
temp = c_features['GarageCars'].replace(np.nan, 0)

c_features['GarageCars'] = temp
test[test['Id']==2490]
c_features.SaleType.value_counts()
temp = c_features['SaleType'].fillna('WD')

c_features['SaleType'] = temp
c_features.isna().sum()
c_features.shape, train_label.shape
c_features.head(3)
c_features.sample(6)
num = c_features.dtypes[c_features.dtypes != "object"].index

c_features[num].skew()
from scipy.special import boxcox1p



print(boxcox1p(c_features['totalArea'], 0.15).skew())

# c_features['totalArea'] = boxcox1p(c_features['totalArea'], 0.15)

c_features['MSSubClass'] = c_features['MSSubClass'].apply(str)



c_features = pd.get_dummies(c_features)

c_features.shape
c_features.head()
# c_features.drop(columns=['Id'], inplace=True)
ntrain = train_label.shape[0]

train_f = c_features[:ntrain]

test_f = c_features[ntrain:]

train_f.shape, test_f.shape
train_f.head()
from sklearn.linear_model import ElasticNet, Lasso

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.001, random_state=99))

enet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.001, random_state=11))

gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_features='sqrt', loss='huber', random_state=5)

rfr = RandomForestRegressor(n_estimators=100, random_state=77)

xgbr = xgb.XGBRegressor(gamma=0.05, learning_rate=0.05, max_depth=3, n_estimators=1200, random_state =100)

lgbr = lgb.LGBMRegressor(objective='regression', learning_rate=0.05, n_estimators=200)
models = [lasso, enet, gbr, rfr, xgbr, lgbr]



for model in models:

    print("cross_val_score: {:.3f}".format(sum(cross_val_score(model, train_f, train_label['SalePrice_log'], cv=5))/5))
chosen_models = [lasso, enet, gbr, xgbr]

class avg_models():

    def __init__(self, models):

        self.models = models

        

    def fit(self, X, y):

        self.models_ = [x for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    def predict(self, X):

        predictions = np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)   
avg = avg_models(models=chosen_models)

avg.fit(train_f, train_label['SalePrice_log'])

avg.predict(test_f)
predictions = avg.predict(test_f)

len(predictions)
len(test_f)
test['SalePrice'] = np.array(np.exp(predictions)-1)



submission = test[['Id', 'SalePrice']]

submission.head(5)
len(test['Id']) ,len(predictions)
sample_submission.head(5)
submission.shape, sample_submission.shape
submission
submission.to_csv('submission4.csv', index=False)