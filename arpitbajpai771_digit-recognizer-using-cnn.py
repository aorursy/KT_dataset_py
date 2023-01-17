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
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

train_data.head()

test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test_data.head()
import tensorflow as tf

tf.__version__
x_test = test_data
y_train = train_data['label']

x_train = train_data.drop(['label'], axis = 1)
g = sns.countplot(y_train)
x_train = tf.keras.utils.normalize(x_train, axis = 1)

x_test = tf.keras.utils.normalize(x_test, axis = 1)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))



model.compile(optimizer = 'adam',

             loss = 'sparse_categorical_crossentropy',

             metrics = ['accuracy'])



model.fit(x_train, y_train, epochs = 3)
results
results = model.predict(x_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")



submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("my_submission_cnn.csv",index=False)