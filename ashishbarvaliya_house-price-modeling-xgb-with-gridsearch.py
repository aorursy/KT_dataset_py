# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train.shape, test.shape
train.head()
train.isna().sum().sort_values()
for col in ['Alley','FireplaceQu','Fence','MiscFeature','PoolQC']:

    train[col].fillna('NA', inplace=True)

    test[col].fillna('NA', inplace=True)

    

train['LotFrontage'].fillna(train["LotFrontage"].value_counts().to_frame().index[0], inplace=True)

test['LotFrontage'].fillna(test["LotFrontage"].value_counts().to_frame().index[0], inplace=True)



train[['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']].isna().head(7)

for col in ['GarageQual','GarageFinish','GarageYrBlt','GarageType','GarageCond']:

    train[col].fillna('NA',inplace=True)

    test[col].fillna('NA',inplace=True)



for col in ['BsmtQual','BsmtCond','BsmtFinType1','BsmtFinType2','BsmtExposure']:

    train[col].fillna('NA',inplace=True)

    test[col].fillna('NA',inplace=True)



train['Electrical'].fillna('SBrkr',inplace=True)



missings = ['GarageCars','GarageArea','KitchenQual','Exterior1st','SaleType','TotalBsmtSF','BsmtUnfSF','Exterior2nd',

            'BsmtFinSF1','BsmtFinSF2','BsmtFullBath','Functional','Utilities','BsmtHalfBath','MSZoning']



numerical=['GarageCars','GarageArea','TotalBsmtSF','BsmtUnfSF','BsmtFinSF1','BsmtFinSF2','BsmtFullBath','BsmtHalfBath']

categorical = ['KitchenQual','Exterior1st','SaleType','Exterior2nd','Functional','Utilities','MSZoning']



# using Imputer class of sklearn libs.

from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median',axis=0)

imputer.fit(test[numerical] + train[numerical])

test[numerical] = imputer.transform(test[numerical])

train[numerical] = imputer.transform(train[numerical])



for i in categorical:

    train[i].fillna(train[i].value_counts().to_frame().index[0], inplace=True)

    test[i].fillna(test[i].value_counts().to_frame().index[0], inplace=True)    



train[train['MasVnrType'].isna()][['SalePrice','MasVnrType','MasVnrArea']]



train[train['MasVnrType']=='None']['SalePrice'].median()

train[train['MasVnrType']=='BrkFace']['SalePrice'].median()

train[train['MasVnrType']=='Stone']['SalePrice'].median()

train[train['MasVnrType']=='BrkCmn']['SalePrice'].median()



train['MasVnrArea'].fillna(181000,inplace=True)

test['MasVnrArea'].fillna(181000,inplace=True)



train['MasVnrType'].fillna('NA',inplace=True)

test['MasVnrType'].fillna('NA',inplace=True)



print(train.isna().sum().sort_values()[-2:-1])

print(test.isna().sum().sort_values()[-2:-1])
int64 =[]

objects = []

for col in train.columns.tolist():

    if np.dtype(train[col]) == 'int64' or np.dtype(train[col]) == 'float64':

        int64.append(col)

    else:

        objects.append(col)                      #here datatype is 'object'



continues_int64_cols = ['LotArea', 'LotFrontage', 'MasVnrArea','BsmtFinSF2','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF',

                  'GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

categorical_int64_cols=[]

for i in int64:

    if i not in continues_int64_cols:

        categorical_int64_cols.append(i)



def barplot(X,Y):

    plt.figure(figsize=(7,7))

    sns.barplot(x=X, y=Y)

    plt.show()

def scatter(X,Y):

    plt.figure(figsize=(7,7))

    sns.scatterplot(alpha=0.4,x=X, y=Y)

    plt.show()

def hist(X):

    plt.figure(figsize=(7,7))

    sns.distplot(X, bins=40, kde=True)

    plt.show()

def box(X):

    plt.figure(figsize=(3,7))

    sns.boxplot(y=X)

    plt.show() 

def line(X,Y):

    plt.figure(figsize=(7,7))    

    sns.lineplot(x=X, y=Y,color="coral")

    plt.show() 



train['MasVnrArea'] = train['MasVnrArea'].apply(lambda row: 1.0 if row>0.0 else 0.0)

train['BsmtFinSF2'] = train['BsmtFinSF2'].apply(lambda row: 1.0 if row>0.0 else 0.0)

binary_cate_int64_cols = []

binary_cate_int64_cols.append('MasVnrArea')

binary_cate_int64_cols.append('BsmtFinSF2')



train['LowQualFinSF'] = train['LowQualFinSF'].apply(lambda row: 1.0 if row>0.0 else 0.0)

binary_cate_int64_cols.append('LowQualFinSF')



for i in continues_int64_cols[14:]:

    train[i] = train[i].apply(lambda row: 1.0 if row>0.0 else 0.0)

    binary_cate_int64_cols.append(i)



for j in binary_cate_int64_cols:

    if j in continues_int64_cols:

        continues_int64_cols.remove(j)        #these special columns removing from the continues_int64_cols



# we changed values of train only, here for test set

for i in binary_cate_int64_cols:

    test[i] = test[i].apply(lambda row: 1.0 if row>0.0 else 0.0)

# test[binary_cate_int64_cols].head(6)



ordinal_categorical_cols =[]

ordinal_categorical_cols.extend(['ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','HeatingQC','KitchenQual'])

ordinal_categorical_cols.extend(['FireplaceQu', 'GarageQual','GarageCond','PoolQC'])



for i in ordinal_categorical_cols:

    if i in objects:

        objects.remove(i)            # removing ordinal features from the objects



# removinf 'Id' and 'SalePrice'

categorical_int64_cols.remove('Id')

categorical_int64_cols.remove('SalePrice')



train['BsmtExposure'] = train['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1,'NA':0})

test['BsmtExposure'] = test['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1,'NA':0})



order = {'Ex':5,

        'Gd':4, 

        'TA':3, 

        'Fa':2, 

        'Po':1,

        'NA':0 }

for i in ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']:

    train[i] = train[i].map(order)

    test[i] = test[i].map(order)

# test[ordinal_categorical_cols].head()   



train_objs_num = len(train)

dataset=pd.concat(objs=[train[categorical_int64_cols+objects+ordinal_categorical_cols],test[categorical_int64_cols+objects+ordinal_categorical_cols]],axis=0)

dataset_preprocessed = pd.get_dummies(dataset.astype(str), drop_first=True)

train_nominal_onehot = dataset_preprocessed[:train_objs_num]

test_nominal_onehot= dataset_preprocessed[train_objs_num:]

# train_nominal_onehot.shape, test_nominal_onehot.shap      



X = pd.concat([train[continues_int64_cols], train[binary_cate_int64_cols], train_nominal_onehot], axis=1)

y = train['SalePrice']

test_final = pd.concat([test[continues_int64_cols], test[binary_cate_int64_cols], test_nominal_onehot], axis=1)

              

X.shape, y.shape, test_final.shape
X.head()
from sklearn import model_selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV



import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor
def xgb_model(X, y, test_final, param, kfolds = 5):

    

    folds = KFold(n_splits=kfolds, shuffle=True, random_state=4590)

    

    feature_importance = np.zeros((X.shape[1],kfolds))

    val_matrix = np.zeros(len(X))

    pred_matrix = np.zeros(len(test_final))



    for fold_, (train_id, val_id) in enumerate(folds.split(X,X)):

        print('--------------Fold-no---',fold_)

        x0,y0 = X.iloc[train_id], y[train_id]

        x1,y1 = X.iloc[val_id], y[val_id]

        

        train_data = xgb.DMatrix(x0, y0)

        val_data = xgb.DMatrix(x1, y1)



        num_round = 10000

        watchlist = [(train_data, 'train'), (val_data, 'valid')]

        

        clf = xgb.train(param, train_data, num_round, watchlist, maximize=False, 

                                            verbose_eval=3000, early_stopping_rounds = 70)



        val_matrix[val_id] = clf.predict(val_data, ntree_limit = clf.best_ntree_limit)

#         feature_importance[:, fold_] = clf.feature_importance()

#         feature_importance[:, fold_] = clf.best_estimator_.booster().get_fscore()

        

        testD = xgb.DMatrix(test_final)

        pred_matrix += clf.predict(testD, ntree_limit = clf.best_ntree_limit) / folds.n_splits

 

    print(' Validation error: ',np.sqrt(mean_squared_error(val_matrix, y)))

    return clf, pred_matrix
params = {

      'objective': 'reg:squarederror', 

      'eval_metric': 'rmse',

      'eta': 0.001,

       'gamma':10,

      'reg_alpha': 0.3,

      'max_depth': 4, 

      'subsample': 0.8,

       'scale_pos_weight': 1,

       'max_bin':256,

#       'best_score': 0.5,

      'tweedie_variance_power': 1.5,

      'colsample_bytree': 1,

      'alpha':0.01,

#        'lambda':90,

       'min_child_weight': 5,

      'random_state': 42, 

      'silent': False}

model, predictions = xgb_model(X ,y , test_final,param = params, kfolds=5)

print("XGBM Training Completed...")
feature_importance = model.get_score(importance_type='gain')

len(feature_importance), feature_importance
features = list(feature_importance.keys())

features[:4]
gridParams = {

      'objective': ['reg:squarederror'], 

      'eval_metric': ['rmse'],

      'eta': [0.001],

       'gamma':[10,20],

      'reg_alpha': [0.3],

      'max_depth': [4,8], 

      'subsample': [0.8],

       'scale_pos_weight': [1],

       'max_bin':[256],

      'tweedie_variance_power': [1.5],

      'colsample_bytree': [0.6,0.8,1],

       'num_round' : [8000],

      'alpha':[0.03,0.01,0.005],

       'min_child_weight': [5,7,9],

      'random_state': [42], 

      'silent': [False]}



clf = xgb.XGBRegressor()



# Create the grid

grid = GridSearchCV(clf, gridParams,

                    verbose=0,

                    cv=3,

                    n_jobs=-1)

# Run the grid

grid.fit(X[features], y)



# Print the best parameters found

print(grid.best_params_)

print(grid.best_score_)
best_params = grid.best_params_

model, predictions = xgb_model(X[features] ,y , test_final[features] ,param = best_params, kfolds=7)

print("XGBM Training Completed...")
submission = pd.DataFrame()

submission['Id'] = test['Id']

submission['SalePrice'] = predictions

submission.to_csv('submission_xgb_gridsearch.csv',index=False)

submission.head()