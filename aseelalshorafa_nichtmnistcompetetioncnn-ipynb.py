import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.image import  imread

import tensorflow as tf 

import numpy as np 

import seaborn as sns 

import os
df = pd.read_csv('../input/nicht-mnist/train.csv',index_col = 0,header=None)
test = pd.read_csv('../input/nicht-mnist/test.csv',index_col = 0,header=None)
test_np = test.values
test_np = test_np / 255
test_np = test_np.reshape(-1, 28, 28 , 1).astype('float32')
df.head()
X = df.drop(columns = [1])

y = df[1]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y)
X_train.shape,X_test.shape


im = np.array(X_train.iloc[50]).astype('uint8').reshape(28,28)

plt.imshow(im)
y.nunique()
y_train
from sklearn.preprocessing import LabelEncoder 

label = LabelEncoder()

y_train = label.fit_transform(y_train)

y_test = label.transform(y_test)
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
y_test
y_train
X_train = X_train.values

X_test = X_test.values
X_train = X_train / 255

X_test = X_test /255
X_train_np = X_train.reshape(-1, 28, 28 , 1).astype('float32')
X_test_np = X_test.reshape(-1, 28, 28 , 1).astype('float32')
X_train_np.shape,X_test_np.shape
from tensorflow.keras.models import Sequential 

from tensorflow.keras.layers import Dense , Conv2D ,MaxPool2D , Flatten
model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(28,28,1), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2)))





model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(28,28,1), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2)))

tf.keras.layers.Dropout(0.2),



model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(28,28,1), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2))) 

tf.keras.layers.Dropout(0.2),



model.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(28,28,1), activation='relu',padding = 'same'))

model.add(MaxPool2D(pool_size=(2, 2))) 

tf.keras.layers.Dropout(0.2),



model.add(Flatten())





model.add(Dense(256, activation='relu'))



model.add(Dense(10, activation='softmax'))





model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping 

early_stopping = EarlyStopping(monitor = 'val_accuracy',patience = 5)

model.fit(X_train_np, y_train, epochs=15, callbacks= [early_stopping],validation_data=(X_test_np,y_test),batch_size = 30)

metrices = pd.DataFrame(model.history.history)
metrices[['accuracy','val_accuracy']].plot()
predictions = model.predict_classes(test_np)
predictions
predictions = label.inverse_transform(predictions)
result = pd.DataFrame({

    'id':test.index,

    'target':predictions}

)
result.to_csv('Nicht MNIST Competetion.csv',index = False)