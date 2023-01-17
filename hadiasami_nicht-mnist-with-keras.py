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
# TensorFlow and tf.keras

import tensorflow as tf

from tensorflow import keras



# Helper libraries

import numpy as np

import matplotlib.pyplot as plt



print(tf.__version__)
train_data = pd.read_csv('../input/nicht-mnist/train.csv',header=None, index_col =0)

test_data = pd.read_csv('../input/nicht-mnist/test.csv',header=None , index_col = 0)

train_data.head()
train_data[1].unique()
y = train_data.pop(1)

x = train_data

x = x / 255.0

test_data = test_data / 255.0
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()

y = pd.DataFrame(label_encoder.fit_transform(y))

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state=1)
from keras.layers import Dense, Dropout


model = keras.models.Sequential([

    keras.layers.Flatten(input_shape=(28, 28)),

    keras.layers.Dense(128, activation='relu'),

    keras.layers.Dense(10)

])
model.compile(  optimizer= 'Adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history = model.fit(x_train, y_train,validation_data=(x_val, y_val), epochs=20)
test_loss, test_acc = model.evaluate(x_val,  y_val, verbose=2)

print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_data)
results = np.argmax(predictions,axis = 1)
results = pd.Series(results,name="target")
submission = pd.concat([pd.Series(range(0,9364),name = "Id"),results],axis = 1)
submission['target'] = label_encoder.inverse_transform(submission['target'])
submission.head(10)
test_out = pd.DataFrame({

    'Id': submission.Id, 

    'target': submission.target

})

test_out.to_csv('submission.csv', index=False)