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

import sklearn.preprocessing

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, make_scorer, roc_curve, auc

import xgboost as xgb

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

import seaborn as sns

sns.set_context("notebook", font_scale=1.1)

sns.set_style("ticks")



train = pd.read_csv('../input/train.csv', index_col = 0)

test = pd.read_csv('../input/test.csv', index_col = 0)
train['yearsToSell'] = train['YrSold'] - train['YearBuilt']

train['yearsSinceRemod'] = train['YrSold'] - train['YearRemodAdd']



test['yearsToSell'] = test['YrSold'] - test['YearBuilt']

test['yearsSinceRemod'] = test['YrSold'] - test['YearRemodAdd']



train['hasBsmt'] = np.where(train['TotalBsmtSF'] > 0, 1, 0)

train['hasPool'] = np.where(train['PoolArea'] > 0, 1, 0)

train['totalArea'] = train['GrLivArea'] + train['GarageArea'] + train['PoolArea']



test['hasBsmt'] = np.where(test['TotalBsmtSF'] > 0, 1, 0)

test['hasPool'] = np.where(test['PoolArea'] > 0, 1, 0)

test['totalArea'] = test['GrLivArea'] + test['GarageArea'] + test['PoolArea']



x = train.drop(labels='SalePrice', axis=1)

y = pd.DataFrame({'SalePrice': train.SalePrice})

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.2)

xtrain.head()
# Using a great pipeline tutorial https://tinyurl.com/yygs8pxb

from sklearn.pipeline import make_pipeline

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



numeric_features = xtrain.select_dtypes(include=['int64','float64']).columns

categorical_features = xtrain.select_dtypes(include=['object']).columns



preprocessor = ColumnTransformer(

    transformers = [

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])
'''# Using this tutorial https://tinyurl.com/y65ulb44 and this notebook https://tinyurl.com/y5xa9ars

from xgboost import XGBRegressor

xgb = XGBRegressor(learning_rate=0.1,n_estimators=5000, early_stopping_rounds=5, eval_set=[(xtest,ytest)])



pipe = Pipeline(steps=[('processor', preprocessor),

                      ('regressor', xgb)])



pipe.fit(xtrain,ytrain)

preds = pipe.predict(test)

df = pd.DataFrame({'Id': test.index, 'SalePrice': preds})

submission = df.to_csv("my_submission.csv", index=False)

Kaggle score was 0.13350'''
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

xgb = XGBRegressor()

'''

param_grid = {

        'silent': [False],

        'max_depth': [6, 10, 15, 20],

        'learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],

        'subsample': [0.4, 0.6, 0.8, 1.0],

        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],

        'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

        'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],

        'n_estimators': [100],

        'objective' : ['reg:squarederror']}





grid_xgb = RandomizedSearchCV(xgb, param_grid, n_iter=20,refit=True, random_state=17)

pipe = Pipeline(steps=[('processor', preprocessor),

                      ('regressor', grid_xgb)])



pipe.fit(x,y)

print("model score: %.3f" % pipe.score(xtest,ytest))

preds = pipe.predict(test)

df = pd.DataFrame({'Id': test.index, 'SalePrice': preds})

submission = df.to_csv("my_submission.csv", index=False)



Kaggle score was 0.14143'''



'''

#https://tinyurl.com/y5onjg4u



from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVC, LinearSVR, NuSVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

regressors = [

    KNeighborsRegressor(10),

    AdaBoostRegressor(),

    NuSVR(),

    RandomForestRegressor(n_estimators=100, max_depth=20)

    ]

for regressor in regressors:

    pipe = Pipeline(steps=[('preprocessor', preprocessor),

                      ('regressor', regressor)])

    pipe.fit(xtrain, ytrain)   

    print(regressor)

    print("model score: %.3f" % pipe.score(xtest, ytest))

    

    RandomForest and NuSVR got poor Kaggle scores

    '''
full = pd.concat([train,test])



full['yearsToSell'] = full['YrSold'] - full['YearBuilt']

full['yearsSinceRemod'] = full['YrSold'] - full['YearRemodAdd']



full['hasBsmt'] = np.where(full['TotalBsmtSF'] > 0, 1, 0)

full['hasPool'] = np.where(full['PoolArea'] > 0, 1, 0)

full['totalArea'] = full['GrLivArea'] + full['GarageArea'] + full['PoolArea']



imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

scaler = StandardScaler()

numeric_features = full.select_dtypes(include=['int64','float64']).columns

categorical_features = full.select_dtypes(include=['object']).columns



# Use the same features and transformations as the training data





full[numeric_features] = imp.fit_transform(full[numeric_features])

full = pd.get_dummies(full)

dummy_cols = full.columns

full = pd.DataFrame(scaler.fit_transform(full))

full.columns = dummy_cols

full.head()
corrmat = full.corr()

corrs = pd.DataFrame(corrmat['SalePrice'].abs().sort_values(ascending=False))

corrs.head(15)
corrmat = full.corr()

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(full[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
cols_to_see = ['SalePrice','totalArea','OverallQual','YearBuilt','yearsToSell']

sns.pairplot(data=full[cols_to_see])