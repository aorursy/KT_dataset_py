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

students_data = pd.read_csv('../input/StudentsPerformance.csv')
students_data.head()
students_data.describe()
students_data.columns
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

"""
I'm not 100% sure how to handle categorical data yet. 
And so I decided to work with whatever numerical values
that's available in the dataset.
"""

"""
Only the scores are available so I decided to use those.
Based on the table above there seems to be a connection between the scores.
I decided to predict the writing score based on the other two scores - math and reading.
"""

y = students_data['writing score']
feature_columns = ['math score', 'reading score']

X = students_data[feature_columns]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
"""
I used both Random Forest and Decision Tree but I found that Random Forest 
provided slighlty better accuracy than Decision.
"""
model = RandomForestRegressor(random_state=1)
# model = DecisionTreeRegressor(random_state=1)

model.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error

preds = model.predict(val_X)
mae = mean_absolute_error(val_y, preds)
print("Mean Absolute Error: {:,.0f}".format(mae))
print("This means that our prediction is around {:,.0f} points away from the actual".format(mae))
print(val_y.head().tolist())
print(preds.tolist())
print(model.score(train_X, train_y))
print(model.score(val_X, val_y))