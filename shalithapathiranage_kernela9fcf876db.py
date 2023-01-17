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
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
import tensorflow as tf
from matplotlib import pyplot as plt
tf.__version__
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
train_data.head()
X = np.array(train_data.drop("label", axis=1)).astype('float32')
y = np.array(train_data['label']).astype('float32')
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(y[i])
plt.show()

X = X / 255.0
X = X.reshape(-1, 28, 28, 1)
y = to_categorical(y)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
X_test = np.array(test_data).astype('float32')
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
plt.figure(figsize=(10,10))
X = np.asarray(X)
print(np.shape(X))
print(np.shape(y))
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(28,28,1)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.1),
tf.keras.layers.Conv2D(32,(3,3),activation = 'relu',input_shape=(28,28,1)),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(units = 64 , activation = 'relu'),
tf.keras.layers.Dropout(0.1),
tf.keras.layers.Dense(units = 16 , activation = 'relu'),
tf.keras.layers.Dropout(0.1),
tf.keras.layers.Dense(10,activation='softmax')])

RMSprop = tf.keras.optimizers.RMSprop(lr=0.001)
model.compile(optimizer='adam' , loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(X,y,epochs = 5, batch_size = 100,validation_split =0.2)
test_data.head()
submit = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submit.shape
submit.Label =model.predict_classes(X_test)

submit.head()
submit.to_csv('submit_3.csv',index=False)
