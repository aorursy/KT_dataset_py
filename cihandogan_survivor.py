import numpy as np 

import pandas as pd



from sklearn.model_selection import train_test_split

import random

import tensorflow as tf

from tensorflow.keras.layers import Dense ,Dropout

from tensorflow.keras.models import Sequential

path = "/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv"

data = pd.read_csv(path)

ds = data.values

random.shuffle(ds)
X,y = (ds[:,4:7],ds[:,7:8])



for i in range(0,len(X)):

    if X[i,0] == 'M':

        X[i,0] = 0

    else:

        X[i,0] = 1

    

    if X[i,2] == 'P':

        X[i,2] = 0

    else:

        X[i,2] = 1

        

#conversion to not get type error.        

X = np.asarray(X).astype(np.float32)

y = np.asarray(y).astype(np.float32)



#test data split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dropout(0.2),

  tf.keras.layers.Dense(10)

])



model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])
history = model.fit(X_train, 

                    y_train, 

                    batch_size= 10, 

                    validation_data=(X_test,y_test))
score = model.evaluate(X, y)