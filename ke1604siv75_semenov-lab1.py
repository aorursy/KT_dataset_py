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

data = pd.read_csv('../input/train.csv')

data.head(10)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(data[['weather','temp','humidity', 'windspeed']], data['count'], test_size=0.33, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)

predict = model.predict(X_test)
predict
from sklearn.metrics import mean_squared_error

from math import sqrt



sqrt(mean_squared_error(y_test, predict))
testData = pd.read_csv('../input/test.csv')

testData.head()
X = testData['weather','temp','humidity', 'windspeed']

y = model.predict(X)

y