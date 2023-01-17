# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class myCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self,epoch,logs={}):

        if(logs.get('acc')>0.6):

            print("model is trained with 70% accuracey")

            self.model.stop_training=true

            

            

mini=tf.keras.datasets.fashion_mnist

(x_train,y_train),(x_test,y_test)=mini.load_data()

model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape=(28, 28)),

  tf.keras.layers.Dense(512, activation=tf.nn.relu),

  tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])

callbacks=myCallback()

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10, callbacks=[callbacks])

score=model.evaluate(

x=x_train,

y=y_train)

print("score",score)
model.predict(x_test)