# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/dataset/train.csv')

values = train.air_pollution_index.values

train.head(5)
train.shape
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder().fit(train.weather_type.values)

train['weather_type'] = le.transform(train.weather_type.values)
train['date_time'] = pd.to_datetime(train.date_time , format = '%Y-%m-%d %H:%M:%S')

data = train.drop(['is_holiday','date_time','air_pollution_index'],axis = 1)

data.index = train.date_time
data.plot(subplots=True,figsize = (10,15) )
dataset = data.values

data_mean = dataset[:33000].mean(axis=0)

data_std = dataset[:33000].std(axis=0)



dataset = (dataset-data_mean)/data_std
def multivariate_data(dataset, target, start_index, end_index, history_size,

                      target_size, step, single_step=False):

  data = []

  labels = []



  start_index = start_index + history_size

  if end_index is None:

    end_index = len(dataset) - target_size



  for i in range(start_index, end_index):

    indices = range(i-history_size, i, step)

    data.append(dataset[indices])



    if single_step:

      labels.append(target[i+target_size])

    else:

      labels.append(target[i:i+target_size])



  return np.array(data), np.array(labels)
past_history = 720

future_target = 72

STEP = 6



x_train_single, y_train_single = multivariate_data(dataset, values, 0,

                                                   33000, past_history,

                                                   future_target, STEP,

                                                   single_step=True)

x_train_single.shape
x_val_single, y_val_single = multivariate_data(dataset, values,

                                               30000, None, past_history,

                                               future_target, STEP,

                                               single_step=True)
x_val_single.shape[-2:]
print ('Single window of past history : {}'.format(x_train_single[0].shape))
import tensorflow as tf

single_step_model = tf.keras.models.Sequential()

single_step_model.add(tf.keras.layers.LSTM(32,

                                           input_shape=(120,11)))

single_step_model.add(tf.keras.layers.Dense(1))



single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')
EVALUATION_INTERVAL = 200

EPOCHS = 10

single_step_history = single_step_model.fit(x_train_single, y_train_single, epochs=EPOCHS,validation_data = (x_val_single, y_val_single))
test = pd.read_csv('/kaggle/input/dataset/test.csv')
test['weather_type'] = le.transform(test.weather_type.values)

date = test['date_time']

test_x = test.drop(['is_holiday','date_time'],axis = 1)
test_x.shape
def multivariate_data_test(dataset,start_index, end_index, history_size,

                      target_size, step, single_step=False):

  data = []



  start_index = start_index + history_size

  if end_index is None:

    end_index = len(dataset) - target_size



  for i in range(start_index, end_index):

    indices = range(i-history_size, i, step)

    data.append(dataset[indices])

  return np.array(data)
dataset = test_x.values

dataset = (dataset-data_mean)/data_std
past_history = 720

future_target = 72

STEP = 6



x_test_single = multivariate_data_test(dataset, 0,

                                                   14454, past_history,

                                                   future_target, STEP,

                                                   single_step=True)
x_test_single.shape