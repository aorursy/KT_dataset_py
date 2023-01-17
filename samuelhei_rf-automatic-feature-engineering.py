# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_squared_log_error,mean_absolute_error
from math import sqrt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

!cat ../input/data_description.txt
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')

def log(n):
    return max(0,np.log(n))

def transform_df(df):
    df['totalAge'] = df['YrSold'] - df['YearBuilt']
    df['AgeRemode'] = df['YrSold'] - df['YearRemodAdd']
    df['totalArea'] = df[[c for c in df.columns if 'Area' in c]].sum(axis=1)
    df['OverallScore'] =  df['OverallQual'] + df['OverallCond']
    
    df.fillna({
        'Alley' : 'No Alley',
        'BsmtQual' : 'No Basement',
        'BsmtCond' : 'No Basement',
        'BsmtExposure' : 'No Basement',
        'BsmtFinType1' : 'No Basement',
        'BsmtFinType2' : 'No Basement',
        'FireplaceQu' : 'No Fireplace',
        'GarageType' : 'No Garage',
        'GarageFinish' : 'No Garage',
        'GarageQual' : 'No Garage',
        'GarageCond' : 'No Garage',
        'PoolQC' : 'No Pool',
        'Fence' : 'No Fence',
        'MiscFeature' : 'None'
    },inplace=True)

    df.fillna(df.mean(), inplace=True)
    df = pd.get_dummies(df)
    return df

y_column = 'SalePrice'
x_columns = list(df.columns)

for c in ['Id','SalePrice']:
    x_columns.remove(c)

X_train, X_test, y_train, y_test = train_test_split(
    df[x_columns], df[y_column], test_size=0.20, random_state=42
)

X_train = transform_df(X_train)
X_test = transform_df(X_test)

columns = []
for c in X_test.columns:
    if c in X_train.columns:
        columns.append(c)
def train_model(columns=X_train.columns, use_log=True, X_train=X_train, X_test=X_test):
    model = RandomForestRegressor(n_estimators=500,random_state=42,n_jobs=-1)
    
    y_train_transformed = y_train.values
    
    if use_log:
        y_train_transformed = np.log(y_train_transformed)
        
    model.fit(X_train[columns], y_train_transformed)
    
    y_pred_train = model.predict(X_train[columns])
    y_pred_test = model.predict(X_test[columns])
    
    if use_log:
        y_pred_train = np.exp(y_pred_train)
        y_pred_test = np.exp(y_pred_test)
    
    train_rms = sqrt(mean_squared_error(np.log(y_train), np.log(y_pred_train)))
    test_rms = sqrt(mean_squared_error(np.log(y_test), np.log(y_pred_test)))
    
    print(train_rms, test_rms)
    
    print('MAE:\t$%.2f' % mean_absolute_error(y_test, y_pred_test))
    print('MSLE:\t%.5f' % mean_squared_log_error(y_test, y_pred_test))

    print()
    
    print(pd.DataFrame(
        {'y_train' : y_train.values,
        'y_pred_train' : y_pred_train}
    ).describe())
    
    print()
    print(pd.DataFrame({
        'y_test' : y_test.values,
        'y_pred_test' : y_pred_test
    }).describe())
    
    return model
model = train_model(columns, use_log=True)
new_columns = list(np.array(columns)[list(np.argsort(model.feature_importances_)[::-1][:100])])
test_model = train_model(new_columns, use_log=True)
new_columns

def pos_process(df):
    new_df = df.copy()

    for c1 in df.columns:
        for c2 in df.columns:
            if c1 == c2:
                continue
                
            name_m =  "_x_".join(sorted([c1,c2]))
            
            if name_m not in new_df.columns:
                new_df[name_m] = df[c1] * df[c2]

    new_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return new_df.fillna(0)
X_train_pos = pos_process(X_train[new_columns])
X_test_pos = pos_process(X_test[new_columns])
pos_model = train_model(X_train_pos.columns, X_train=X_train_pos, X_test=X_test_pos, use_log=True)
new_columns_pos = list(np.array(X_train_pos.columns)[list(np.argsort(pos_model.feature_importances_)[::-1][:80])])
pos_model_ = train_model(new_columns_pos, X_train=X_train_pos, X_test=X_test_pos, use_log=False)
new_columns_pos
!head ../input/sample_submission.csv
submission = pd.read_csv('../input/test.csv')
test = transform_df(submission.copy())
test = pos_process(test)
submission['SalePrice'] = pos_model_.predict(test[new_columns_pos])
submission[['Id','SalePrice']]
submission[['Id','SalePrice']].to_csv('rf.csv',index=False)