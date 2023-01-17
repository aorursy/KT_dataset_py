# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#path of file to read
training_file_path = '../input/pubg-finish-placement-prediction/train_V2.csv'

X = pd.read_csv(training_file_path)

X.dropna(axis=0, subset=['winPlacePerc'], inplace=True)
y = X.winPlacePerc
X.drop(['winPlacePerc'], axis=1, inplace=True)

#drop categorical data
X = X.select_dtypes(exclude=['object'])
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
dt_model = RandomForestRegressor(n_jobs=-1, n_estimators = 25, max_leaf_nodes=10000, random_state=1)
dt_model.fit(train_X, train_y)
val_predictions = dt_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.4f}".format(val_mae))
final_model = RandomForestRegressor(n_jobs=-1, n_estimators = 25, max_leaf_nodes=10000, random_state=1)
final_model.fit(X, y)
# path to file you will use for predictions
test_data_path = '../input/pubg-finish-placement-prediction/test_V2.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# test_X which comes from test_data but includes only the columns used for prediction.
#drop categorical data
test_X = test_data.select_dtypes(exclude=['object'])

# make predictions which we will submit. 
test_preds = final_model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'winPlacePerc': test_preds})
output.to_csv('submission.csv', index=False)