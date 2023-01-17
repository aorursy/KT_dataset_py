import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import os



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor



from xgboost import XGBRegressor

def num_missing(data):

    ms_counts = data.isnull().sum()

    return ms_counts[ms_counts!=0]
raw_train = pd.read_csv('../input/train.csv')

raw_test = pd.read_csv('../input/test.csv')
raw_train.head()
raw_test.head()
raw_train.describe()
raw_train.info()
raw_train.select_dtypes(['float64','int64']).columns
raw_train.select_dtypes(['object']).columns
plt.figure(num=None, figsize=(12, 10))

sns.heatmap(raw_train.corr())
raw_train.corr().SalePrice.sort_values(ascending=False)
missing = num_missing(raw_train)

missing
raw_train[missing.index.values].dtypes
train = raw_train.copy()

test = raw_test.copy()
train['MSSubClass'] = train['MSSubClass'].astype('object')
cat_columns = train.select_dtypes('object').columns
train[cat_columns] = train[cat_columns].fillna('missing')
test[cat_columns] = test[cat_columns].fillna('missing')
train = pd.get_dummies(train)

test = pd.get_dummies(test)

train, test = train.align(test,join='left',axis=1)
train_X = train.drop(columns=['Id','SalePrice'])

train_y = train.SalePrice

test_X = test.drop(columns=['Id','SalePrice'])
rnreg = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('reg', RandomForestRegressor(n_estimators=500,max_depth=3))

])
rnreg.fit(train_X,train_y);
feat_importances = rnreg['reg'].feature_importances_
impdf = pd.DataFrame({'Feature Name':train_X.columns,'Importance':feat_importances})
impdf.sort_values('Importance',ascending=False).head(15).plot(x='Feature Name',y='Importance',kind='barh')
model = Pipeline([

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('reg', XGBRegressor(objective='reg:squarederror',n_estimators=2000,max_depth=20))

])
from sklearn.model_selection import cross_val_score



np.sqrt(-cross_val_score(model, train_X,train_y, cv=5,scoring='neg_mean_squared_error'))
model.fit(train_X, train_y)
test_preds = model.predict(test_X)



output = pd.DataFrame({'Id':test.Id,'SalePrice':test_preds})

output.to_csv('submission.csv',index=False)