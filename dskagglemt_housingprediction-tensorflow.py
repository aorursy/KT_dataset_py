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
hosuing = pd.read_csv('/kaggle/input/housingprice-prediction/Housing_Price.csv')
hosuing.head()
hosuing.describe().transpose()
X = hosuing.drop(['medianHouseValue'], axis = 1)

X.head()
y = hosuing.medianHouseValue

y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train.head()
X_train = pd.DataFrame(data = scaler.transform(X_train), columns = X_train.columns, index = X_train.index)
X_train.head()
X_test = pd.DataFrame(data = scaler.transform(X_test), columns = X_test.columns, index = X_test.index)
import tensorflow as tf
age         = tf.compat.v1.feature_column.numeric_column('housingMedianAge')

rooms       = tf.compat.v1.feature_column.numeric_column('totalRooms')

bedrooms    = tf.compat.v1.feature_column.numeric_column('totalBedrooms')

pop         = tf.compat.v1.feature_column.numeric_column('population')

households  = tf.compat.v1.feature_column.numeric_column('households')

income      = tf.compat.v1.feature_column.numeric_column('medianIncome')
feat_cols = [age, rooms, bedrooms, pop, households, income]
input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_train, y = y_train, batch_size = 10, num_epochs = 1000, shuffle = True)
model = tf.compat.v1.estimator.DNNRegressor(hidden_units = [6,6,6], feature_columns = feat_cols)
model.train(input_fn = input_func, steps = 25000)
predict_input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x = X_test, batch_size = 10, num_epochs = 1, shuffle = False)
pred_gen = model.predict(predict_input_func)
predictions = list(pred_gen)
predictions
final_pred = []



for pred in predictions:

    final_pred.append(pred['predictions'])
final_pred
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, final_pred)**0.5

mse