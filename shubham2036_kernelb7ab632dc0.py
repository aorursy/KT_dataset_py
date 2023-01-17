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
dataset = pd.read_csv('../input/mydataset/final_data.csv')

dataset = dataset.set_index('shot_id_number')

dataset1 = dataset[['power_of_shot','knockout_match','distance_of_shot','area_of_shot','shot_basics','range_of_shot']]

dataset1 = pd.get_dummies(dataset1)

dataset1.info()
dataset1['is_goal'] = dataset['is_goal']
dataset1.head()
# x = dataset1['distance_of_shot'].mean()
# x
# dataset1['distance_of_shot'] = dataset1['distance_of_shot']/x
# dataset1['distance_of_shot'].mean()
submission = pd.read_csv('../input/zs-associate/sample_submission.csv')
submission.info()
testing = dataset1.loc[submission['shot_id_number']]
testing.info()
training = dataset1.dropna()

dataset = training.iloc[np.random.permutation(len(training))]

X = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1]

y=y.astype('int64')

y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train.info()
testing = testing.iloc[:,:-1]
import tensorflow as tf

from tensorflow import keras
model = keras.Sequential([

    keras.layers.Input(21),

    keras.layers.BatchNormalization(),

    keras.layers.Dense(64, activation=tf.nn.relu,use_bias = True,kernel_regularizer= keras.regularizers.l1(0.001)),

     keras.layers.Dense(64, activation=tf.nn.relu,use_bias = True,kernel_regularizer= keras.regularizers.l1(0.001)),

    keras.layers.BatchNormalization(),

#     keras.layers.Dropout(0.3),

     keras.layers.Dense(16, activation=tf.nn.relu,use_bias = True,kernel_regularizer= keras.regularizers.l1(0.001)),

    keras.layers.Dense(16, activation=tf.nn.relu,use_bias = True,kernel_regularizer= keras.regularizers.l1(0.001)),

    keras.layers.BatchNormalization(),

#     keras.layers.Dropout(0.3),

    keras.layers.Dense(4, activation=tf.nn.relu,use_bias = True,kernel_regularizer= keras.regularizers.l1(0.001)),

    keras.layers.Dense(4, activation=tf.nn.relu,use_bias = True,kernel_regularizer= keras.regularizers.l1(0.001)),

#     keras.layers.Dense(4, activation=tf.nn.relu),

    

    keras.layers.Dense(1,activation = tf.nn.relu)

])


model.compile(optimizer='adam',

              loss=tf.keras.losses.binary_crossentropy,

              metrics=['accuracy'])
model.fit(X,y,epochs=250)
y_pred = model.predict(X_test)

y_pred =(y_pred>0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
y_pred=model.predict(testing)

submission['is_goal'] = y_pred

submission.to_csv(path_or_buf='shubham_raj_031298_prediction_5.csv')