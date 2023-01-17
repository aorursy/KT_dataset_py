import tensorflow as tf

import pandas as pd
import numpy as np

from keras.constraints import MinMaxNorm
from keras import backend as K



def root_mean_squared_error(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))*100
data_frame = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
data_frame
X_training = data_frame['date'].str.split('.')[:20000]

prices = data_frame['item_price'][:20000]

for i in range(len(X_training)):

    X_training[i] = [int(item) for item in X_training[i]]

    X_training[i].append(prices[i])

X_training = np.array([np.array(item).reshape(4) for item in X_training])
X_training.shape
X_test = data_frame['date'].str.split('.')[40000:50000].to_numpy()

prices = data_frame['item_price'][40000:50000].to_numpy()

for i in range(len(X_test)):

    X_test[i] = [int(item) for item in X_test[i]]

    X_test[i].append(prices[i])

X_test = np.array([np.array(item).reshape(4) for item in X_test])
Y_train = data_frame['item_cnt_day'][:20000]
Y_test = data_frame['item_cnt_day'][40000:50000]
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=4, activation='relu', input_shape=(4,),use_bias=True, bias_constraint=

                               MinMaxNorm(min_value=0.5, max_value=1.0, rate=1.0, axis=0)

))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units=1))

model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=['mean_absolute_error'])
model.summary()
model.fit(X_training, Y_train, epochs=10)
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
import pandas as pd

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")

sample_submission = pd.read_csv("../input/competitive-data-science-predict-future-sales/sample_submission.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")

test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")