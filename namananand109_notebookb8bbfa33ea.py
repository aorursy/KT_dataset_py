
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/CarPrice_Assignment.csv')
pd.set_option('display.max_columns', None)

data.head()
## step 1 is split the independent and dependent variables from each other
X= data.drop('price',axis=1)
y=data.price
X.head()
data_types =X.dtypes
data_types

columns_to_removed = data_types.loc[data_types==object].index
new_X = X.drop(columns_to_removed ,axis=1)
new_X.head()
train_x = new_X.head(200)
test_x = new_X.tail(5)
train_y = y.head(200)
test_y = y.tail(5)
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(train_x , train_y)
reg.coef_
train_x.columns
test_x
prediction = reg.predict(test_x)
test_y
from sklearn.metrics import mean_squared_error
import math
math.sqrt(mean_squared_error(test_y, prediction))