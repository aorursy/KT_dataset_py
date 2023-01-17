

import warnings

warnings.filterwarnings("ignore")



import seaborn as sns

import numpy as np 

import pandas as pd

import matplotlib.pylab as plt



%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
voice = pd.read_csv('../input/voicegender/voice.csv')
voice.head()
#Printing the total distribution



print("Total Number of samples : {}".format(voice.shape[0]))



print("Total No.of Males : {}".format(voice[voice.label == 'male'].shape[0]))



print("Total No.of Females : {}".format(voice[voice.label == 'female'].shape[0]))

#Checking for Null Values



voice.isnull().sum()
sns.pairplot(voice[['meanfreq', 'Q25', 'Q75', 'skew', 'centroid', 'label']], 

                 hue='label', size=2)
voice.head()

voice.plot(kind='scatter', x='meanfreq', y='dfrange')

voice.plot(kind='kde', y='meanfreq')
voice.label = [1 if each == "male" else 0 for each in voice.label]



voice.head()
X = voice.drop(["label"], axis = 1)

y = voice.label.values



#Normalizing X



X = (X - X.min()) / (X.max() - X.min())



print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)



y_train = y_train.reshape(-1,1)

y_test = y_test.reshape(-1,1)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
import tensorflow as tf

import tensorflow.keras as keras
model = keras.Sequential([keras.layers.InputLayer(input_shape=X_train.shape[1:]),

                          keras.layers.Dense(32, activation='relu'),

                          keras.layers.Dense(64, activation='relu'),

                          keras.layers.Dropout(0.2),

                          keras.layers.Dense(32, activation='relu'),

                          keras.layers.Dropout(0.2),

                          keras.layers.Dense(16, activation='relu'),

                          keras.layers.Dense(1, activation='sigmoid')

                         ])
model.summary()
model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,

                              patience=5, min_lr=0.001)



v_split= 0.2

epoch = 100
history = model.fit(X_train, y_train, validation_split = v_split , epochs =epoch, callbacks=[reduce_lr])
history_df = pd.DataFrame(history.history)



history_df.plot(figsize=(10,10))



plt.grid(True)

plt.gca().set_ylim(0,1.15)

plt.show()
pred = model.evaluate(X_test, y_test)