# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

train_data=pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.dropna(axis=0,subset=['SalePrice'],inplace=True)

target=train_data.SalePrice

# Run this code block with the control-enter keys on your keyboard. Or click the blue botton on the left

cols_with_missing=[col for col in train_data.columns
                          if train_data[col].isnull().any()]
candidate_train_predictors=train_data.drop(['Id','SalePrice']+cols_with_missing,axis=1)
candidate_test_predictors=test_data.drop(['Id']+cols_with_missing,axis=1)

#Removing high cardinality columns
low_cardinality_cols=[cname for cname in candidate_train_predictors.columns
                             if candidate_train_predictors[cname].nunique()<10 and
                                candidate_train_predictors[cname].dtype=="object"]
numeric_cols=[cname for cname in candidate_train_predictors.columns
                     if candidate_train_predictors[cname].dtype in ['int64','float64']]
my_cols=numeric_cols + low_cardinality_cols
train_predictors=candidate_train_predictors[my_cols]
test_predictors=candidate_test_predictors[my_cols]

one_hot_encoded_training_predictors=pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors=pd.get_dummies(test_predictors)
final_train,final_test=one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,join='inner',axis=1)
#print(final_test.isnull().sum())

def get_mae(X,y):
    return -1*cross_val_score(RandomForestRegressor(50),X,y,scoring='neg_mean_absolute_error').mean()

predictors_without_categoricals=train_predictors.select_dtypes(exclude=['object'])
mae_without_categoricals=get_mae(predictors_without_categoricals,target)
mae_one_hot_encoding=get_mae(final_train,target)
print("Mean absolute error when dropping categoricals:")
print(mae_without_categoricals)
print("Mean absolute error with one hot encoding:")
print(mae_one_hot_encoding)

iowa_model=RandomForestRegressor()
iowa_model.fit(final_train,target)

my_imputer=SimpleImputer()
imputed_final_test=my_imputer.fit_transform(final_test)

predicted_prices=iowa_model.predict(imputed_final_test)
print("Some predicted prices are:",predicted_prices)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
