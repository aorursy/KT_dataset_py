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
#importing needed libraries

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

import statsmodels.api as sm

import statsmodels.formula.api as smf

%matplotlib inline



col_list = ['#cc615c', '#6965a7', '#f1bdbf']

sns.set_palette(col_list)
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train.info()
test.info()
train.describe()
test.describe()
def missing_zero_values_table(df):

        zero_val = (df == 0.00).astype(int).sum(axis=0)

        mis_val = df.isnull().sum()

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)

        mz_table = mz_table.rename(

        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})

        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']

        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)

        mz_table['Data Type'] = df.dtypes

        mz_table = mz_table[

            mz_table.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      

            "There are " + str(mz_table.shape[0]) +

              " columns that have missing values.")

#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)

        return mz_table



missing_zero_values_table(train)
missing_zero_values_table(test)
#dropping columns with more than 40% of missing values

train = train.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1)

test = test.drop(['PoolQC','MiscFeature','Alley','Fence','FireplaceQu'], axis=1)
#imputing for null values

#to impute the variables with median for numerical columns

train = train.fillna(train['LotFrontage'].value_counts().index[0])

train = train.fillna(train['GarageYrBlt'].value_counts().index[0])

train = train.fillna(train['MasVnrArea'].value_counts().index[0])



#to impute the variables with median for numerical columns

test = test.fillna(train['LotFrontage'].value_counts().index[0])

test = test.fillna(train['GarageYrBlt'].value_counts().index[0])

test = test.fillna(train['MasVnrArea'].value_counts().index[0])
#checking for null values

#(train.isna().sum()/len(train))*100

#(test.isna().sum()/len(test))*100
train.columns[train.isnull().any()]

test.columns[test.isnull().any()]
# Total number of rows and columns

train.shape

# Rows containing duplicate data

duplicate_rows_df = train[train.duplicated()]

#print('number_of duplicate rows:'+ duplicate_rows_df.shape)

#number of duplicate rows:  (989, 10)

duplicate_rows_df
# Total number of rows and columns

test.shape

# Rows containing duplicate data

duplicate_rows_df_test = test[test.duplicated()]

#print('number_of duplicate rows:'+ duplicate_rows_df.shape)

#number of duplicate rows:  (989, 10)

duplicate_rows_df_test
#changing the datatypes

train.select_dtypes(include=['object']).columns
test.select_dtypes(include=['object']).columns
train.MSZoning.unique()
## converting Categorical Data into proper forms except account_info

list1 =['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition']



for col in list1:

    train[col] = train[col].astype('category')



### from category to int labels

for col in list1:

    train[col] = train[col].cat.codes
## converting Categorical Data into proper forms except account_info

list1 =['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition']



for col in list1:

    test[col] = test[col].astype('category')



### from category to int labels

for col in list1:

    test[col] = test[col].cat.codes
#viewing the dataset with the help of PROFILER

#import pandas_profiling



#pandas_profiling.ProfileReport(train)

#pandas_profiling.ProfileReport(test)



#extracting profiler report in html 

#profile1 = pandas_profiling.ProfileReport(master_data_final)

#profile1.to_file(outputfile="Trade_Report_India.html")
train.columns
X=train[['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',

       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',

       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',

       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',

       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',

       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',

       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',

       'Functional', 'Fireplaces', 'GarageType', 'GarageYrBlt', 'GarageFinish',

       'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive',

       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']]

y= train[['SalePrice']]
#train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
#feature importance

import numpy as np

import matplotlib.pyplot as plt



#from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesRegressor



# Build a forest and compute the feature importances

forest = ExtraTreesRegressor(n_estimators=250,random_state=0)



forest.fit(X, y)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(0,20):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))





# Plot the feature importances of the forest

feat_importances = pd.Series(importances, index=X.columns)

feat_importances.nlargest(20).plot(kind='barh',label=True)
train.columns
X=train[['BedroomAbvGr', 'GarageType','CentralAir','LotArea','YearRemodAdd','TotRmsAbvGrd','BsmtFinSF1',

         'Fireplaces','2ndFlrSF','BsmtQual','1stFlrSF','GarageArea','YearBuilt','TotalBsmtSF','FullBath','KitchenQual',

         'GrLivArea','ExterQual','GarageCars','OverallQual'

       ]]

y=train[['SalePrice']]
X1=train[['BedroomAbvGr', 'GarageType','CentralAir','LotArea','YearRemodAdd','TotRmsAbvGrd','BsmtFinSF1',

         'Fireplaces','2ndFlrSF','BsmtQual','1stFlrSF','GarageArea','YearBuilt','TotalBsmtSF','FullBath',

          'KitchenQual','GrLivArea','ExterQual','GarageCars','OverallQual']]

y1=train[['SalePrice']]
test_new=test[['BedroomAbvGr', 'GarageType','CentralAir','LotArea','YearRemodAdd','TotRmsAbvGrd','BsmtFinSF1','1stFlrSF',

       'GarageArea','YearBuilt','TotalBsmtSF','FullBath','KitchenQual','GrLivArea','ExterQual','GarageCars','OverallQual',

       'Fireplaces','2ndFlrSF','BsmtQual']]
#Using XGBoost

import xgboost as xgb

from sklearn.metrics import mean_squared_error

data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(train_X,train_y,eval_set=[(train_X, train_y), (test_X, test_y)],eval_metric='logloss',verbose=True)



preds = xg_reg.predict(test_X)
evals_result = xg_reg.evals_result()

evals_result
rmse = np.sqrt(mean_squared_error(test_y, preds))

print("RMSE: %f" % (rmse))
#k-fold Cross Validation using XGBoost

params = {"objective":"reg:squarederror",'colsample_bytree': 0.3,'learning_rate': 0.1,

                'max_depth': 5, 'alpha': 10}



cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="mae", as_pandas=True, seed=123)
cv_results.head()
print((cv_results["test-mae-mean"]).tail(1))
#Visualize Boosting Trees and Feature Importance

xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
import matplotlib.pyplot as plt



xgb.plot_tree(xg_reg,num_trees=0)

plt.rcParams['figure.figsize'] = [200, 50]

plt.show()
#test_y
preds_df=pd.DataFrame(preds,columns=['SalePrice'],index=None)

preds_df
#preds_df.shape
#test_y.shape
import neptune
import datetime

import lightgbm as lgb

import numpy as np

import os

import pandas as pd

import random

from tqdm import tqdm

from sklearn.model_selection import train_test_split

import haversine
feature_names = X.columns.tolist()



#test train split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=20)

X_train.shape



#creating lightGBM dataset

lgtrain = lgb.Dataset(X_train, y_train,

                feature_name=feature_names)

lgvalid = lgb.Dataset(X_test, y_test,

                feature_name=feature_names)
#LightGBM Hyperparameters + early stopping

gbm = lgb.LGBMRegressor(learning_rate = 0.15, metric = 'l1', 

                        n_estimators = 28,boosting_type='gbdt', objective='regression',max_bin=1000000)





gbm.fit(X_train, y_train,

        eval_set=[(X_test, y_test)],

        eval_metric=['auc', 'l1'],

early_stopping_rounds=5)
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_0 = user_secrets.get_secret("api_token")
secret_value_0
import neptune



neptune.init('priya-pathak/sandbox',

             api_token=secret_value_0)



neptune.create_experiment()

    

def neptune_monitor():

    def callback(env):

        for name, loss_name, loss_value, _ in env.evaluation_result_list:

            neptune.send_metric('{}_{}'.format(name, loss_name), x=env.iteration, y=loss_value)

    return callback
params = {

    'objective' : 'regression_l1',

    'boosting':'dart',

    'metric' : 'rmse',

    'num_leaves' : 300,

    'max_depth': 20,

    'learning_rate' : 0.1,

    'feature_fraction' : 0.8,

    'verbosity' : 1,

    'num_iteratios':10000,

    'num_threads':2,

    'lambda_l1': 3.097758978478437,

    'lambda_l2': 2.9482537987198496,

    'min_child_weight': 6.996211413900573,

    'min_split_gain': 0.037310344962162616

    

    

}

lgb_clf = lgb.train(

    params,

    lgtrain,

    num_boost_round = 1000,

    valid_sets=[lgtrain, lgvalid],

    valid_names=["train", "valid"],

    early_stopping_rounds=1000,

    callbacks=[neptune_monitor()]

)



print("MAPE of the validation set:", np.sqrt(mean_squared_error(y_test, lgb_clf.predict(X_test))))

fig, ax = plt.subplots(figsize=(10, 7))

lgb.plot_importance(lgb_clf, max_num_features=30, ax=ax)

plt.title("LightGBM - Feature Importance");
#predicting data

y_pred = gbm.predict(test_new,num_iteration=gbm.best_iteration_)

y_pred
preds_df1=pd.DataFrame(y_pred,columns=['SalePrice'],index=None)

preds_df1
test_y
test1=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

test1
submission = pd.concat([test1[['Id']], pd.DataFrame(preds_df1,columns=['SalePrice'])], axis=1)

#submission=preds_df1

submission.to_csv('submission_lightGBM4.csv', index=False)
submission