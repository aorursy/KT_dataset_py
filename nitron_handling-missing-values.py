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
train_path = '../input/train.csv'

train_data = pd.read_csv(train_path)

train_data.head()
test_path = '../input/test.csv'

test_data = pd.read_csv(test_path)

test_data.head()
miss_col = (train_data.isnull().sum())

print(miss_col[miss_col>0])
cols_with_missing = [col for col in train_data.columns if train_data[col].isnull().any()]

dcols_train = train_data.drop(cols_with_missing,axis=1)

dcols_test = test_data.drop(cols_with_missing,axis=1)



dcols_train.head()
from sklearn.impute import SimpleImputer



# imputes numeric data and leaves rest as it is

def impute_numeric(data):

    my_imputer = SimpleImputer()



    int_data = data.select_dtypes('int64')

    float_data =data.select_dtypes('float64')

    numeric_data=pd.concat([int_data,float_data],axis=1)



    num_imputed = my_imputer.fit_transform(numeric_data)

    num_imputed = pd.DataFrame(np.column_stack(list(zip(*num_imputed))),columns=numeric_data.columns)



    data_imputed = data.copy()

    data_imputed.update(num_imputed)

    return data_imputed
# impute numeric in train data

data_imputed = impute_numeric(train_data)



# find missing values left from non numeric

imp_miss = [col for col in data_imputed.columns if data_imputed[col].isnull().any()]

# drop from train, test imputed frames

data_imputed = data_imputed.drop(imp_miss,axis=1)

test_imputed = impute_numeric(test_data.drop(imp_miss,axis=1))

data_imputed.head()
data_ext = train_data.copy()

test_ext = test_data.copy()



# add col_missing to train and test data

cols_miss = [col for col in data_ext.columns if data_ext[col].isnull().any()]

for col in cols_miss:

    data_ext[col+"_missing"]=data_ext[col].isnull()

    test_ext[col+"_missing"]=test_ext[col].isnull()



data_ext = impute_numeric(data_ext)



# drop non numeric missing data

cols_miss = [col for col in data_ext.columns if data_ext[col].isnull().any()]

data_ext = data_ext.drop(cols_miss,axis=1)

test_ext = impute_numeric(test_data.drop(cols_miss,axis=1))

data_ext.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error



def get_error(data,features):

    model = RandomForestRegressor(random_state=1)



    X=data[features]

    y=data['SalePrice']

    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=1) 



    model.fit(train_X,train_y)

    val_predictions = model.predict(val_X)

    return mean_absolute_error(val_predictions, val_y)
def get_numeric_cols(data):

    int_cols = data.select_dtypes("float64")

    float_cols = data.select_dtypes("int64")

    num_cols = pd.concat([int_cols,float_cols],axis=1)

    return num_cols.columns.values
features = get_numeric_cols(dcols_train.drop(["SalePrice"],axis=1))

get_error(dcols_train,features)
features=get_numeric_cols(data_imputed.drop(['SalePrice'],axis=1))



get_error(data_imputed,features)
features=get_numeric_cols(data_ext.drop(['SalePrice'],axis=1))

bool_features=data_ext.select_dtypes("bool").columns.values

features=np.concatenate((features,bool_features))

get_error(data_ext,features)