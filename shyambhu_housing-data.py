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
data_train=pd.read_csv('../input/train.csv')

print('training shape is:',data_train.shape)

data_test=pd.read_csv('../input/test.csv')

print('test shape is:',data_test.shape)

cols=list(data_train.columns)

print(cols)
string_cols = []

for col in cols: 

    col_meaning = data_train[~data_train[col].isna()][col].tolist()

    type_col = type(col_meaning[0]).__name__ 

    if type_col == 'str' or type_col == 'object':

        string_cols.append(col)
'Alley' in string_cols
data_train['flag'] = 'train'

data_test['flag'] = 'test'

data = pd.concat([data_train,data_test])
def add_dummies(data):

    small_data = pd.DataFrame()

    for col in string_cols:

        curr_data = pd.get_dummies(data[col],prefix = col)

        data = data.drop(col,axis = 1)

        small_data = pd.concat([small_data,curr_data], axis = 1)

        print(col,small_data.shape)

    data = pd.concat([data,small_data],axis = 1)

    return data

data = add_dummies(data)
cols = list(data.columns)

for col in cols:

    if type(data[col][0]).__name__ == 'str':

        print(col)

data.dtypes
data_train = data[data['flag']=='train']

data_test = data[data['flag'] == 'test']

data_train = data_train.drop(['flag'],axis = 1)

data_test = data_test.drop(['flag'],axis = 1)



print(data_train.shape)

print(data_test.shape)

cols_train = set(data_train.columns)

cols_test = set(data_test.columns)

#print(data_train.columns)

#print(data_test.columns)

print(len(cols_train.intersection(cols_test)))
cols = list(data_train.columns)

#for col in cols:

#    print(col)

num_cols = ['TotRmsAbvGrd','3SsnPorch','BedroomAbvGr','GarageArea',

            'GrLivArea','LotArea','GarageYrBlt','PoolArea']

for col in num_cols:

    data_train[col] = data_train[col].apply(lambda x: np.log(x+1))

for col in num_cols:

    print(data_train[col].describe())
#linear regression

from sklearn.ensemble import RandomForestRegressor as rfreg

from sklearn.metrics import mean_squared_error as rmse_1

from sklearn.impute import SimpleImputer

my_imputer=SimpleImputer()

data_train = data_train[~(data_train['SalePrice'].isna())]

X_train = data_train.drop('SalePrice',axis = 1)

Y_train = data_train['SalePrice']

X_test = data_test

X_train=my_imputer.fit_transform(X_train)

X_test=my_imputer.fit_transform(X_test)



linreg=rfreg(n_estimators = 1000, max_depth = 8,min_samples_split = 15, max_features = 95,

             oob_score=True,n_jobs=-1)

fitted_model = linreg.fit(X_train,Y_train)

predictions=fitted_model.predict(X_test)

pred_train = fitted_model.predict(X_train)

print(rmse_1(pred_train,Y_train)**0.5)

from sklearn.metrics import r2_score as rs

print(rs(Y_train,pred_train))

print(fitted_model.oob_score_)
Y_train.describe()
predictions_table=pd.DataFrame()

predictions_table['Id'] = [1461+i for i in range(1459)]

predictions_table['SalePrice'] = predictions

predictions_table.to_csv('vanilla_log_submission.csv',index = False)