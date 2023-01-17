# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Import libraries

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

dataset_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
dataset_train.head()
dataset_test.head()
dataset_test.info()
dataset_test.describe()
missing = dataset_train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
dataset_train.shape
dataset_train['PoolQC'].isnull().sum(),dataset_train['Alley'].isnull().sum(),dataset_train['MiscFeature'].isnull().sum()
dataset_train.drop(['Alley'],axis=1,inplace=True)

dataset_train.drop(['MiscFeature'],axis =1,inplace=True)

dataset_train.drop(['PoolQC'],axis=1,inplace=True)

dataset_train.drop(['Fence'],axis=1,inplace=True)
dataset_train.shape
dataset_train['GarageFinish']=dataset_train['GarageFinish'].fillna(dataset_train['GarageFinish'].mode()[0])

dataset_train['GarageQual']=dataset_train['GarageQual'].fillna(dataset_train['GarageQual'].mode()[0])

dataset_train['GarageCond']=dataset_train['GarageCond'].fillna(dataset_train['GarageCond'].mode()[0])

dataset_train['GarageType']=dataset_train['GarageType'].fillna(dataset_train['GarageType'].mode()[0])

dataset_train['LotFrontage']=dataset_train['LotFrontage'].fillna(dataset_train['LotFrontage'].mean())

dataset_train['BsmtCond']=dataset_train['BsmtCond'].fillna(dataset_train['BsmtCond'].mode()[0])

dataset_train['BsmtQual']=dataset_train['BsmtQual'].fillna(dataset_train['BsmtQual'].mode()[0])
dataset_train.drop(['GarageYrBlt'],axis=1,inplace=True)
dataset_train.drop(['Fireplaces','FireplaceQu'],axis=1,inplace=True)
dataset_train.shape
dataset_test.shape

dataset_test.drop(['Alley','MiscFeature','PoolQC','Fence','FireplaceQu','Fireplaces','GarageYrBlt'],axis=1,inplace=True)

dataset_test['LotFrontage']=dataset_test['LotFrontage'].fillna(dataset_test['LotFrontage'].mean())

dataset_test['MSZoning']=dataset_test['MSZoning'].fillna(dataset_test['MSZoning'].mode()[0])

dataset_test['BsmtCond']=dataset_test['BsmtCond'].fillna(dataset_test['BsmtCond'].mode()[0])

dataset_test['BsmtQual']=dataset_test['BsmtQual'].fillna(dataset_test['BsmtQual'].mode()[0])

dataset_test['GarageType']=dataset_test['GarageType'].fillna(dataset_test['GarageType'].mode()[0])

dataset_test['GarageFinish']=dataset_test['GarageFinish'].fillna(dataset_test['GarageFinish'].mode()[0])

dataset_test['GarageQual']=dataset_test['GarageQual'].fillna(dataset_test['GarageQual'].mode()[0])

dataset_test['GarageCond']=dataset_test['GarageCond'].fillna(dataset_test['GarageCond'].mode()[0])
dataset_test['BsmtExposure']=dataset_test['BsmtExposure'].fillna(dataset_test['BsmtExposure'].mode()[0])

dataset_test['BsmtFinType2']=dataset_test['BsmtFinType2'].fillna(dataset_test['BsmtFinType2'].mode()[0])

dataset_test['MasVnrType']=dataset_test['MasVnrType'].fillna(dataset_test['MasVnrType'].mode()[0])

dataset_test['MasVnrArea']=dataset_test['MasVnrArea'].fillna(dataset_test['MasVnrArea'].mode()[0])

dataset_test['Utilities']=dataset_test['Utilities'].fillna(dataset_test['Utilities'].mode()[0])

dataset_test['Exterior1st']=dataset_test['Exterior1st'].fillna(dataset_test['Exterior1st'].mode()[0])

dataset_test['Exterior2nd']=dataset_test['Exterior2nd'].fillna(dataset_test['Exterior2nd'].mode()[0])

dataset_test['BsmtFinType1']=dataset_test['BsmtFinType1'].fillna(dataset_test['BsmtFinType1'].mode()[0])

dataset_test['BsmtFinSF1']=dataset_test['BsmtFinSF1'].fillna(dataset_test['BsmtFinSF1'].mean())

dataset_test['BsmtFinSF2']=dataset_test['BsmtFinSF2'].fillna(dataset_test['BsmtFinSF2'].mean())

dataset_test['BsmtUnfSF']=dataset_test['BsmtUnfSF'].fillna(dataset_test['BsmtUnfSF'].mean())

dataset_test['TotalBsmtSF']=dataset_test['TotalBsmtSF'].fillna(dataset_test['TotalBsmtSF'].mean())

dataset_test['BsmtFullBath']=dataset_test['BsmtFullBath'].fillna(dataset_test['BsmtFullBath'].mode()[0])

dataset_test['BsmtHalfBath']=dataset_test['BsmtHalfBath'].fillna(dataset_test['BsmtHalfBath'].mode()[0])

dataset_test['KitchenQual']=dataset_test['KitchenQual'].fillna(dataset_test['KitchenQual'].mode()[0])

dataset_test['Functional']=dataset_test['Functional'].fillna(dataset_test['Functional'].mode()[0])

dataset_test['GarageCars']=dataset_test['GarageCars'].fillna(dataset_test['GarageCars'].mean())

dataset_test['GarageArea']=dataset_test['GarageArea'].fillna(dataset_test['GarageArea'].mean())

dataset_test['SaleType']=dataset_test['SaleType'].fillna(dataset_test['SaleType'].mode()[0])
import xgboost

classifier=xgboost.XGBRegressor()

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
## Hyper Parameter Optimization

n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
from sklearn.model_selection import RandomizedSearchCV

random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
dataset_test.shape
final_df=pd.concat([dataset_train,dataset_test],axis=0)
final_df['SalePrice']
def category_onehot_multcols(multcolumns):

    df_final=final_df

    i=0

    for fields in multcolumns:

        

        print(fields)

        df1=pd.get_dummies(final_df[fields],drop_first=True)

        

        final_df.drop([fields],axis=1,inplace=True)

        if i==0:

            df_final=df1.copy()

        else:

            

            df_final=pd.concat([df_final,df1],axis=1)

        i=i+1

       

        

    df_final=pd.concat([final_df,df_final],axis=1)

        

    return df_final



columns=['MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',

         'Condition2','BldgType','Condition1','HouseStyle','SaleType',

        'SaleCondition','ExterCond',

         'ExterQual','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',

        'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Heating','HeatingQC',

         'CentralAir',

         'Electrical','KitchenQual','Functional',

         'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive']
final_df=category_onehot_multcols(columns)
final_df.shape


final_df =final_df.loc[:,~final_df.columns.duplicated()]
Train=final_df.iloc[:1422,:]

Test=final_df.iloc[1422:,:]
Train.head()
Test.head()
Test.drop(['SalePrice'],axis=1,inplace=True)
X_train=Train.drop(['SalePrice'],axis=1)

y_train=Train['SalePrice']
import xgboost

classifier=xgboost.XGBRegressor()

regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']

base_score=[0.25,0.5,0.75,1]
n_estimators = [100, 500, 900, 1100, 1500]

max_depth = [2, 3, 5, 10, 15]

booster=['gbtree','gblinear']

learning_rate=[0.05,0.1,0.15,0.20]

min_child_weight=[1,2,3,4]



# Define the grid of hyperparameters to search

hyperparameter_grid = {

    'n_estimators': n_estimators,

    'max_depth':max_depth,

    'learning_rate':learning_rate,

    'min_child_weight':min_child_weight,

    'booster':booster,

    'base_score':base_score

    }
random_cv = RandomizedSearchCV(estimator=regressor,

            param_distributions=hyperparameter_grid,

            cv=5, n_iter=50,

            scoring = 'neg_mean_absolute_error',n_jobs = 4,

            verbose = 5, 

            return_train_score = True,

            random_state=42)
random_cv.fit(X_train,y_train)
random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,

       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,

       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,

       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

       silent=True, subsample=1)

regressor.fit(X_train,y_train)
import pickle

filename = 'finalized_model.pkl'

pickle.dump(classifier, open(filename, 'wb'))
Test.drop(['SalePrice'],axis=1,inplace=True)
y_pred=regressor.predict(df_Test.drop(['SalePrice'],axis=1))
y_pred
pred=pd.DataFrame(ann_pred)

sub_df=pd.read_csv('sample_submission.csv')

datasets=pd.concat([sub_df['Id'],pred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)