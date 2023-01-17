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
train_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/train.csv")

test_data = pd.read_csv("/kaggle/input/home-data-for-ml-course/test.csv")

test=test_data
train_data.head()
test_data.head()
Col_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

print(Col_with_missing)
Col_with_missing = [col for col in test_data.columns if test_data[col].isnull().any()]

print(Col_with_missing)
drop_column=[]
for col in train_data.columns:

    if train_data[col].isnull().any():

        x=train_data[col].isnull().sum()

        if x>50:

            print(col+"\t"+str(x))

            drop_column.append(col)
for col in test_data.columns:

    if test_data[col].isnull().any():

        x=test_data[col].isnull().sum()

        if x>50:

            print(col+"\t"+str(x))
print(drop_column)
train_data=train_data.drop(drop_column,axis=1)

test_data=test_data.drop(drop_column,axis=1)
X_train=train_data.drop(["SalePrice"],axis=1)

Y_train=train_data["SalePrice"]
X_train.head()
Y_train.head()
cat_column=[]
for col in X_train.columns:

    if X_train[col].dtype=='object':

        cat_column.append(col)
print(cat_column)
for col in cat_column:

    print(col)

    print(X_train[col].value_counts())

    print("-"*50)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



# function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.01)

    my_model.fit(X_train, y_train, early_stopping_rounds=50, 

             eval_set=[(X_valid, y_valid)], verbose=False)

    my_model.fit(X_train, y_train)

    preds = my_model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)

x_test=test_data
from sklearn.impute import SimpleImputer

my_imputer=SimpleImputer(strategy="most_frequent")

imputed_X_train= pd.DataFrame(my_imputer.fit_transform(x_train))

imputed_X_test=pd.DataFrame(my_imputer.transform(x_test))

imputed_X_valid=pd.DataFrame(my_imputer.transform(x_valid))

imputed_X_train.index = x_train.index

imputed_X_valid.index = x_valid.index

imputed_X_test.index = x_test.index

imputed_X_train.columns=x_train.columns

imputed_X_valid.columns=x_valid.columns

imputed_X_test.columns=x_test.columns
Col_with_missing_2 = [col for col in imputed_X_valid.columns if imputed_X_valid[col].isnull().any()]

print(Col_with_missing_2)
Num_col=[]
for col in x_train.columns:

    if(x_train[col].dtype!="object"):

        print(col+"\t"+str(x_train[col].dtype))

        if col!="Id":

            Num_col.append(col)
print(Num_col)
#feature=["LotArea","OverallQual","OverallCond","BsmtUnfSF","TotalBsmtSF","1stFlrSF","GrLivArea",'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',"GarageArea"]

"""

feature=[]

for col in Num_col:

    feature.append(col)

"""

feature=['TotalBsmtSF', 'WoodDeckSF', 'BsmtUnfSF', 'YearRemodAdd', '3SsnPorch', 'KitchenAbvGr', '2ndFlrSF', 'ScreenPorch', 'PoolArea', 'TotRmsAbvGrd', 'MoSold', 'BedroomAbvGr', 'MiscVal', 'BsmtHalfBath', '1stFlrSF', 'GarageCars', 'OverallQual', 'YrSold', 'HalfBath', 'OpenPorchSF', 'BsmtFullBath', 'LowQualFinSF', 'LotArea', 'OverallCond', 'YearBuilt', 'EnclosedPorch', 'FullBath', 'Fireplaces', 'BsmtFinSF2', 'BsmtFinSF1', 'MSSubClass', 'GrLivArea', 'GarageArea']
print(feature)
for col in cat_column:

    print(col)

    print(X_train[col].value_counts())

    print("-"*50)
#cat_enc_col=["Street","LotShape","LotConfig","BldgType","HouseStyle","MasVnrType","ExterQual","Foundation","BsmtQual","BsmtExposure","BsmtFinType1","KitchenQual"]

cat_enc_col=['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']
for col in cat_enc_col:

    feature.append(col)
imputed_X_train=imputed_X_train[feature]

imputed_X_valid=imputed_X_valid[feature]

imputed_X_test=imputed_X_test[feature]
imputed_X_train.head(10)
Cat_cols=cat_enc_col
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_X_train[Cat_cols]))

OH_cols_valid = pd.DataFrame(OH_encoder.transform(imputed_X_valid[Cat_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(imputed_X_test[Cat_cols]))

OH_cols_train.index = imputed_X_train.index

OH_cols_valid.index = imputed_X_valid.index

OH_cols_test.index = imputed_X_test.index

num_X_train = imputed_X_train.drop(Cat_cols, axis =1)

num_X_valid = imputed_X_valid.drop(Cat_cols, axis =1)

num_X_test = imputed_X_test.drop(Cat_cols, axis =1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)

OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
OH_X_train = OH_X_train.apply(pd.to_numeric)

OH_X_valid = OH_X_valid.apply(pd.to_numeric)

OH_X_test = OH_X_test.apply(pd.to_numeric)
OH_X_test.head(10)
print("MAE:",end=" ")

print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
from sklearn.ensemble import GradientBoostingRegressor

my_model = GradientBoostingRegressor(loss="ls",learning_rate=0.01,n_estimators=1000,max_depth=4,alpha=0.08)

my_model.fit(OH_X_train, y_train)

preds = my_model.predict(OH_X_valid)

print("MAE:",mean_absolute_error(y_valid, preds))
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error

my_model_1 = XGBRegressor(n_estimators=2000, learning_rate=0.008)

my_model_1.fit(OH_X_train, y_train, early_stopping_rounds=50, 

             eval_set=[(OH_X_valid, y_valid)], verbose=False)

preds_1 = my_model_1.predict(OH_X_valid)

print("MAE:",mean_absolute_error(y_valid, preds_1))
from sklearn.ensemble import GradientBoostingRegressor

my_model_2 = GradientBoostingRegressor(loss="ls",learning_rate=0.01,n_estimators=2000,max_depth=4,alpha=0.08)

my_model_2.fit(OH_X_train, y_train)

preds_2 = my_model_2.predict(OH_X_valid)

print("MAE:",mean_absolute_error(y_valid, preds_2))
preds_3=(preds_1+preds_2)/2

print("MAE:",mean_absolute_error(y_valid, preds_3))
predictions1 = my_model_1.predict(OH_X_test)

predictions2 = my_model_2.predict(OH_X_test)

Preds_last=(predictions1+predictions2)/2
Preds_last
output = pd.DataFrame({'Id': test.Id,'SalePrice': Preds_last})

output.to_csv('submission1.csv', index=False)