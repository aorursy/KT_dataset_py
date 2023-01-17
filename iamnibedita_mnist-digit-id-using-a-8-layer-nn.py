# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
y = df['label']

df = df.drop(['label'],axis=1)





X = df.to_numpy()

X_test_test = df_test.to_numpy()

print(X.shape)

X = X.reshape(42000,28,28)

X_test_test = X_test_test.reshape(28000,28,28)



df = df/255.0

df_test = df_test/255.0
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
callbacks = keras.callbacks.EarlyStopping(monitor='sparse_categorical_accuracy', mode=min)

model = keras.Sequential([keras.layers.Flatten(),

                         keras.layers.Dense(1024,activation=tf.nn.relu),

                         keras.layers.Dense(512,activation=tf.nn.relu),

                         keras.layers.Dense(256,activation=tf.nn.relu),

                          keras.layers.Dense(128,activation=tf.nn.relu),

                          keras.layers.Dense(64,activation=tf.nn.relu),

                          keras.layers.Dense(32,activation=tf.nn.relu),

                          keras.layers.Dense(16,activation=tf.nn.relu),

                         keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),

              loss=keras.losses.SparseCategoricalCrossentropy(),

              metrics=[keras.metrics.SparseCategoricalAccuracy()]

             )



print ('Fit model on training data')

model.fit(X_train,y_train,epochs=25,callbacks=[callbacks])



print ('Evaluate on test data')

result = model.evaluate(X_test,y_test)



print ('Make predictions using model')

predictions_proba = model.predict(X_test_test)

prediction_classes = predictions_proba.argmax(axis=-1)

print (prediction_classes)
print ('Make predictions using model')

predictions_proba = model.predict(X_test_test)

prediction_classes = predictions_proba.argmax(axis=-1)

print (prediction_classes)
submit=pd.DataFrame(columns=['ImageId','Label'])



submit['Label'] = prediction_classes

submit['ImageId'] = submit.reset_index().index + 1

submit.to_csv('sample_submission.csv', index=False)
