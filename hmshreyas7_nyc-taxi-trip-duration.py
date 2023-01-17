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
taxi_trip_data = pd.read_csv('../input/train.csv')
taxi_trip_data.describe()
X = taxi_trip_data.iloc[:, 5:9]
y = taxi_trip_data.trip_duration

X.head()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state = 1, n_estimators = 100)
model.fit(X, y)
test = pd.read_csv('../input/test.csv')
test_X = test.iloc[:, 4:8]

test_X.head()
predictions = model.predict(test_X)
submission = pd.DataFrame({'id': test.id, 'trip_duration': predictions})
submission.to_csv('submission.csv', index = False)