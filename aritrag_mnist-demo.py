# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import tensorflow as tf

from tensorflow import keras

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

print(train_df.shape)
X_train = train_df.iloc[:,1:].values

y_train = train_df.iloc[:,0].values
N = X_train.shape[0]

X_train = X_train.reshape(N,28,28,1)

X_train = X_train.astype('float32')

X_train /= 255.



y_train = tf.one_hot(y_train, 10)

X_train.shape, y_train.shape
def make_model(inputs):

    x = tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3))(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    # 26,26,32

    x = tf.keras.layers.MaxPool2D()(x)

    # 12,12,32

    x = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3))(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    # 10,10,64

    x = tf.keras.layers.MaxPool2D()(x)

    # 5,5,64

    x = tf.keras.layers.Conv2D(filters=1024,kernel_size=(5,5))(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    #1,1,1024

    x = tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1))(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.ReLU()(x)

    #1,1,256

    x = tf.keras.layers.Conv2D(filters=10,kernel_size=(1,1))(x)

    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Softmax()(x)

    #1,1,10

    x = tf.keras.layers.Reshape((10,))(x)

    return x
tf.keras.backend.clear_session()

inputs = tf.keras.Input(shape=(28,28,1))



outputs = make_model(inputs)



model = tf.keras.Model(

    inputs=inputs,

    outputs=outputs,

    name="simple"

)



model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()
history = model.fit(X_train,y_train, validation_split=0.2,epochs=10)
X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv').values



N_test = X_test.shape[0]

X_test = X_test.reshape(N_test,28,28,1)

X_test = X_test.astype('float32')

X_test /= 255.



X_test.shape
pred = model.predict(X_test)

pred.shape
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),

                         "Label": np.argmax(pred, axis=1)})

submissions.to_csv("my_submissions.csv", index=False, header=True)