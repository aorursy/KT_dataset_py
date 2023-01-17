import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist

import os
print(os.listdir("../input"))

%matplotlib inline
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_train.head()
x_train = np.array(df_train.iloc[:,1:])
x_train = np.array([np.reshape(i, (28, 28, 1)) for i in x_train])
y_train = np.array(df_train.iloc[:,0])
x_train = x_train/255.0
y_train = keras.utils.to_categorical(y_train)
x_test = np.array(df_test)
x_test = np.array([np.reshape(i, (28, 28, 1)) for i in x_test])
x_test = x_test/255.0
# 6 random plots

img_indices = [random.randint(0,33600) for i in range(6)] 
n=0
fig = plt.figure(figsize=[15,10])
axes = fig.subplots(2, 3)
for row in range(2):
    for col in range(3):
        axes[row,col].imshow((x_train[img_indices[n]]).reshape((28,28)), cmap='Accent')
        n += 1
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='random_uniform', padding='same', activation='relu', input_shape=(X_train.shape[1:])))
model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=64, kernel_size=(5,5), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(filters=128, kernel_size=(7,7), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.Conv2D(filters=128, kernel_size=(7,7), kernel_initializer='random_uniform', padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(3,3)))

model.add(keras.layers.Conv2D(filters=256, kernel_size=(7,7), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))


model.add(keras.layers.Dense(units=256, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=y_train.shape[1], activation='softmax'))


print(model.summary())

model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
filepath = "model.h5"
ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
rlp = ReduceLROnPlateau(monitor='loss', patience=2, factor=0.2)
# Configure and train the model
history = model.fit(X_train, Y_train,  batch_size=500, callbacks=[es, ckpt, rlp], epochs=100, validation_data=(X_test,Y_test))
# 40 random plots to test our model

img_indices = [random.randint(0,1000) for i in range(40)]
n=0
fig = plt.figure(figsize=[30,50])
axes = fig.subplots(10, 4)
for row in range(10):
    for col in range(4):
        axes[row,col].imshow((X_test[img_indices[n]][:,:,0]).reshape((28,28)), cmap='Accent')
        predicted_num = np.argmax(model.predict(X_test[img_indices[n]:img_indices[n]+1]))
        actual_num = np.argmax(Y_test[img_indices[n]])
        axes[row,col].set_title("{}. Predicted = {} | Actual = {}".format(n+1, predicted_num, actual_num), fontsize=15)
        n += 1
id_img = []
label = []
for i in range(len(x_test)):
    id_img.append(i+1)
    label.append(np.argmax(model.predict(x_test[i:i+1])))
    
img_id = np.array(id_img)
label = np.array(label)
op_df = pd.DataFrame()
op_df['ImageId'] = img_id
op_df['Label'] = label
op_df.to_csv("submission.csv", index=False)