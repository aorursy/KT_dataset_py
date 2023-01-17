import pandas as pd

import numpy as np

from scipy.stats import skew

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

import seaborn as sns

import warnings

def ingnore_warn(*args, **kwargs):

    pass

warnings.warn = ingnore_warn

%pylab inline
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
train_data.head()
test_data.head()
train_data.drop(['Id'],axis=1, inplace = True)

test_data.drop(['Id'],axis=1, inplace = True)
train_data.shape, test_data.shape
train_data.describe()
train_data.isnull().sum().sort_values(ascending=False, inplace=False)
test_data.isnull().sum().sort_values(ascending=False, inplace=False)
train_data['SalePrice'].hist(bins=45)

print(skew(train_data.SalePrice))
train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
y = train_data['SalePrice']

train_data.drop(['SalePrice'], axis = 1, inplace = True)
train_data.shape, test_data.shape
all_data = pd.concat([train_data, test_data], axis=0)
all_data.shape
categorial_features = [prizn for prizn in train_data.columns if all_data[prizn].dtype.name == 'object']

numeric_features = [prizn for prizn in train_data.columns if all_data[prizn].dtype.name != 'object']

print(categorial_features, numeric_features, sep = '\n')
all_data[categorial_features].describe()
all_data[numeric_features].describe()
all_data = all_data.fillna(all_data.mean(axis=0))
all_data[numeric_features].count(axis=0)
for one_prizn in categorial_features:

    all_data[one_prizn] = all_data[one_prizn].fillna('None')

    
all_data[categorial_features].count(axis=0)
all_data = pd.get_dummies(all_data)
all_data.columns
all_data[numeric_features] = StandardScaler().fit_transform(all_data[numeric_features])
X_train = all_data[:train_data.shape[0]]

X_test = all_data[train_data.shape[0]:]

print(X_train.shape, X_test.shape, y.shape)
X_train, X_test, y
kfolds = KFold(n_splits = 7, shuffle = True, random_state = 11)





def rmse_cv(model):

    return np.sqrt(-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv= kfolds))

    

    
from sklearn import ensemble

import xgboost as xgb
rfregr = ensemble.RandomForestRegressor(n_estimators=300, max_depth = 3, random_state = 5)

rfregr.fit(X_train, y)
rmse_cv(rfregr).mean()
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

model_xgb.fit(X_train, y)
rmse_cv(model_xgb).mean()
print('Predict submission')

submission = pd.read_csv("sample_submission.csv")

submission.iloc[:,1] = (np.expm1(model_xgb.predict(X_test)))

print(submission)
submission.to_csv('../input/my_submission.csv', index = False)