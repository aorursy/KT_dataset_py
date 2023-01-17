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
#read training data into a dataframe

home_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
#import sklearn models

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



#fill NA data with 0s

home_df.fillna(value=0)
#create target object and label as y

y = home_df.SalePrice



#create feature list and label as X

features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'BsmtFinSF1', 'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'GarageCars', 'MiscVal']

X = home_df[features]



#split data into validation data and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



#specify RandomForestRegressor model

rf_model = RandomForestRegressor(random_state=1)



#fit model

rf_model.fit(train_X, train_y)



#predict against validation data and calculate mean absolute error

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions,val_y)

print(rf_val_mae)



# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)



#read test data into dataframe

test_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_data_noNA = test_data.fillna(value=0)



#create text_X from features list

test_X=test_data_noNA[features]



#predict against test data

test_preds = rf_model_on_full_data.predict(test_X)



#save submission data 

output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
