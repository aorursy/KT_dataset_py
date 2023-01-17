# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

# .iloc[:,1:] drops labels and retains just pixel data

y, x = data['label'].values, data.iloc[:,1:].values.astype(float)/255

# Reshaping for tf batch optimization

x = x.reshape(-1,28,28,1)
test = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

x_test = test.iloc[:,1:].values.astype(float)/255
dig = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')

x_dig = dig.iloc[:,1:].values.astype(float)/255

x_dig = x_dig.reshape(-1,28,28,1)

y_dig = dig.label.values
x_test = x_test.reshape(-1,28,28,1)
from sklearn.preprocessing import OneHotEncoder
#Transform target into onehot

ohe = OneHotEncoder(handle_unknown='ignore')

y = ohe.fit_transform(y[:,np.newaxis])
import tensorflow as tf

#Make sure to utilize the gpu though this shouldn't be necessary

tf.device('/GPU:0')
#Build model. Construction is on the heavy side but because competition only takes one forward pass it shouldn't matter. Doesn't close in on the time limit

model = tf.keras.Sequential([tf.keras.layers.Conv2D(input_shape=(28,28,1),activation ='relu', filters = 621, kernel_size = 3, padding='same'),

                             tf.keras.layers.Dropout(0.4),

                             tf.keras.layers.Conv2D(488,3,padding = 'same'),

                             tf.keras.layers.Dropout(0.4),

                             tf.keras.layers.Conv2D(388,3,padding = 'same'),

                             tf.keras.layers.Dropout(0.4),

                             tf.keras.layers.Conv2D(276,3,padding = 'same'),

                             tf.keras.layers.Dropout(0.4),

                             tf.keras.layers.Flatten(),

                             tf.keras.layers.Dense(300,activation = 'relu'),

                             tf.keras.layers.Dropout(0.4),

                             tf.keras.layers.Dense(10, activation = 'softmax')

    

])
#Compile the model. Adam for stochastic gradient descent.

model.compile(optimizer = 'adam',

             loss = 'categorical_crossentropy',

             metrics = ['acc'])
#Fit model. Usually takes only couple epochs to find solution. Epochs are overdone to allow early stopping to do its magic.

model.fit(x, y,

          epochs = 100,

          validation_split = 0.2,

          callbacks = [tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', mode = 'max', patience = 3, restore_best_weights = True)])
#Test model against the "actual" test set "dig"

dig_pred = tf.keras.backend.eval(tf.argmax(model.predict(x_dig),1))

np.sum(dig_pred==y_dig)/y_dig.shape[0]
#Make competition predictions and write the result to csv

pred = tf.keras.backend.eval(tf.argmax(model.predict(x_test),1))

out = pd.DataFrame({'id': test['id'], 'label': pred})

out.to_csv("submission.csv", index=False)