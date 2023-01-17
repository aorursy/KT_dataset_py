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
import tensorflow as tf

from os import path, getcwd, chdir
path = f"{getcwd()}/../input/digit-recognizer"



import tensorflow as tf



class CustomCallbacks(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):

        if(logs.get('accuracy')>0.99):

            print("\n 99% acc reached")

            self.model.stop_training = True





mnist = tf.keras.datasets.mnist



(x_train, y_train),(x_test, y_test) = mnist.load_data()



x_train = x_train / 255

x_test = x_test / 255



model = tf.keras.models.Sequential([

    tf.keras.layers.Flatten(input_shape=(28,28)),

    tf.keras.layers.Dense(units=128, activation=tf.nn.relu),

    tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(x_train,y_train,epochs=10, callbacks=[CustomCallbacks()]

)
cp=model.predict(x_test)

cp
import matplotlib.pyplot as plt



plt.imshow(x_test[10])
plt.imshow(x_test[0])
model.evaluate(x_test,y_test)
model.summary()