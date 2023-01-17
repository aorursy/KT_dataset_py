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
# Read the data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

# Do the one hot encoding
enc_train_data = pd.get_dummies(train_data)
enc_test_data =pd.get_dummies(test_data)
final_train, final_test = enc_train_data.align(enc_test_data,join='left',axis=1)

# Define input and outputs
y = final_train['SalePrice']
X = final_train.drop(['SalePrice'],axis=1)


# Train and val split
from sklearn.model_selection import train_test_split

train_X,val_X,train_y,val_y = train_test_split(X.as_matrix(),y.as_matrix(),test_size=0.25)

# Fill the missing values using imputer
from sklearn.preprocessing import Imputer
my_imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)

imp_train_X = my_imputer.fit_transform(train_X)
imp_val_X = my_imputer.transform(val_X)
final_test = final_test.drop(['SalePrice'],axis=1)
imp_test_X = my_imputer.transform(final_test)


# Create XG boost model
from xgboost import XGBRegressor

my_model = XGBRegressor()
my_model.fit(imp_train_X,train_y,verbose=False)


# Predict values
predictions = my_model.predict(imp_val_X)

# Calculate the mae
from sklearn.metrics import mean_absolute_error
print("Mean absolute error : " + str(mean_absolute_error(predictions, val_y)))


# Fit the model by tuning the params
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(imp_train_X,train_y,early_stopping_rounds=5,eval_set=[(imp_val_X,val_y)], verbose=False)

predictions = my_model.predict(imp_val_X)
print("Mean absolute error  after tuning: " + str(mean_absolute_error(predictions, val_y)))

# Submission predictions

sub_predictions = my_model.predict(imp_test_X)
my_submission = pd.DataFrame({ 'Id': final_test.Id,'SalePrice': sub_predictions})
my_submission.to_csv('submission.csv',index=False)

