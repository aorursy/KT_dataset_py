# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
train_Id = train["Id"]

test_Id = test["Id"]

#storing train and test id to use in future for prediction
train.describe()
test.head()
test.describe()
sns.heatmap(train.isnull())
sns.heatmap(test.isnull())
train.info()
test.info()
train.isnull().sum().sort_values(ascending=False)[0:20]
test.isnull().sum().sort_values(ascending=False)[0:35]
#deleting those columns which have more than 50% Nan values

#as those columns are same for both test and train datas

list_drop=["PoolQC","MiscFeature","Alley","Fence","GarageYrBlt"]



for col in list_drop:

    del train[col]

    del test[col]
train.isnull().sum().sort_values(ascending=False)[0:15]
test.isnull().sum().sort_values(ascending=False)[0:30]
train.LotFrontage.value_counts(dropna=False)
train.LotFrontage.fillna(train.LotFrontage.mean(),inplace=True)

test.LotFrontage.fillna(test.LotFrontage.mean(),inplace=True)
print(train.BsmtCond.value_counts(dropna=False))

print(test.BsmtCond.value_counts(dropna=False))
list_fill_train=["BsmtCond", "BsmtQual", "GarageType", "GarageCond", "GarageFinish",

                 "GarageQual","MasVnrType","BsmtFinType2","BsmtExposure","FireplaceQu","MasVnrArea"]



for j in list_fill_train:

    #df_train[j].fillna(df_train[j].mode(),inplace=True)

    # wrong way to do it.

    train[j] = train[j].fillna(train[j].mode()[0])

    test[j] = test[j].fillna(train[j].mode()[0])
print(train.isnull().sum().sort_values(ascending=False)[0:5])

print(test.isnull().sum().sort_values(ascending=False)[0:20])
train.dropna(inplace=True)
train.shape
list_test_str = ['BsmtFinType1', 'Utilities','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'SaleType', 'Exterior2nd', 

           'Exterior1st', 'KitchenQual','MSZoning']

list_test_num= ['GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',  'TotalBsmtSF', 'GarageArea',]



for item in list_test_str:

    test[item] = test[item].fillna(test[item].mode()[0])

for item in list_test_num:

    test[item] = test[item].fillna(test[item].mean())
print(train.isnull().sum().sort_values(ascending=False)[0:5])

print(test.isnull().sum().sort_values(ascending=False)[0:5])
test.shape
Y=train["SalePrice"]
del train["Id"]

del test["Id"]

del train["SalePrice"]
test.shape
train.shape
#joining data sets

final=pd.concat([train,test],axis=0)
final.shape
columns = ['MSZoning', 'Street','LotShape', 'LandContour', 'Utilities',

           'LotConfig', 'LandSlope','Neighborhood', 'Condition1', 'Condition2',

           'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

           'Exterior2nd', 'MasVnrType','ExterQual', 'ExterCond', 'Foundation',

           'BsmtQual', 'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

           'Heating', 'HeatingQC', 'CentralAir', 'Electrical','KitchenQual',

           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish',

           'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
def One_hot_encoding(columns):

    final_df=final

    i=0

    for fields in columns:

        df1=pd.get_dummies(final[fields],drop_first=True)

        

        final.drop([fields],axis=1,inplace=True)

        if i==0:

            final_df=df1.copy()

        else:           

            final_df=pd.concat([final_df,df1],axis=1)

        i=i+1

       

        

    final_df=pd.concat([final,final_df],axis=1)

        

    return final_df
final = One_hot_encoding(columns)
final.head()
final.shape
#changing columns names to unique value

cols = []

count = 1

for column in final.columns:

    cols.append(count)

    count+=1

    continue

    

final.columns = cols
from sklearn import preprocessing

# Get column names first

names = final.columns

# Create the Scaler object

scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object

scaled_df = scaler.fit_transform(final)

final = pd.DataFrame(scaled_df, columns=names)
df_train=final.iloc[:1422,:]

df_test=final.iloc[1422:,:]

df_test.shape
X = df_train

X.shape
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)

print(X_train.shape,X_test.shape)

print(Y_train.shape,Y_test.shape)
linear_reg=LinearRegression()

linear_reg.fit(X_train,Y_train)
print("R-Squared Value for Training Set: {:.3f}".format(linear_reg.score(X_train,Y_train)))

print("R-Squared Value for Test Set: {:.3f}".format(linear_reg.score(X_test,Y_test)))
R_forest=RandomForestRegressor()

R_forest.fit(X_train,Y_train)
print("R-Squared Value for Training Set: {:.3f}".format(R_forest.score(X_train,Y_train)))

print("R-Squared Value for Test Set: {:.3f}".format(R_forest.score(X_test,Y_test)))
y_pred_rforest = R_forest.predict(df_test)
import xgboost
regressor=xgboost.XGBRegressor()
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,

       colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,

       importance_type='gain', interaction_constraints='',

       learning_rate=0.1, max_delta_step=0, max_depth=2,

       min_child_weight=1, monotone_constraints='()',

       n_estimators=900, n_jobs=0, num_parallel_tree=1,

       objective='reg:squarederror', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method='exact',

       validate_parameters=1, verbosity=None)
regressor.fit(X_train,Y_train)
print("R-Squared Value for Training Set: {:.3f}".format(regressor.score(X_train,Y_train)))

print("R-Squared Value for Test Set: {:.3f}".format(regressor.score(X_test,Y_test)))
y_pred_xgb = regressor.predict(df_test)
y_pred_xgb
y_pred_01 = (y_pred_rforest + y_pred_xgb)/2
pred_df_01 = pd.DataFrame(y_pred_01, columns=['SalePrice'])

test_id_df = pd.DataFrame(test_Id, columns=['Id'])
submission_01 = pd.concat([test_id_df, pred_df_01], axis=1)
submission_01.to_csv(r'submission_ens_01.csv', index=False)
submission_01.head()
y_pred_02=(0.86*y_pred_rforest + 0.90*y_pred_xgb)/(0.86+0.90)
pred_df_02 = pd.DataFrame(y_pred_02, columns=['SalePrice'])

test_id_df = pd.DataFrame(test_Id, columns=['Id'])
submission_02 = pd.concat([test_id_df, pred_df_02], axis=1)
submission_02.to_csv(r'submission_ens_02.csv', index=False)
submission_02.head()