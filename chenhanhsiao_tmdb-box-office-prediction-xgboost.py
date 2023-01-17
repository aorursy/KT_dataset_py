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
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer



data = pd.read_csv('../input/train.csv')

print(data.columns)

print(data.head())

data.dropna(axis=0, subset=['revenue'], inplace=True)

y = data.revenue

X = data.drop(['revenue'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)



print("X describe")

print(pd.DataFrame(data=X).describe())

print("y describe")

print(pd.DataFrame(data=y).describe())



print(X.columns)

features = X.columns



my_imputer = Imputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)
from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_X, train_y, verbose=False)
train_predictions = my_model.predict(train_X)

test_predictions = my_model.predict(test_X)



print(pd.DataFrame(data=train_predictions).describe())

print(pd.DataFrame(data=test_predictions).describe())

print(pd.DataFrame(data=train_y).describe())

print(pd.DataFrame(data=test_y).describe())



from sklearn.metrics import mean_squared_log_error

print("training Root-Mean-Squared-Logarithmic-Error (RMSLE) : " + str(np.sqrt(mean_squared_log_error(train_predictions, train_y))))

print("testing  Root-Mean-Squared-Logarithmic-Error (RMSLE) : " + str(np.sqrt(mean_squared_log_error(test_predictions, test_y))))
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

print(test_data.columns)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = my_model.predict(test_X.values)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'id': test_data.id,

                       'revenue': test_preds})

output.to_csv('submission.csv', index=False)