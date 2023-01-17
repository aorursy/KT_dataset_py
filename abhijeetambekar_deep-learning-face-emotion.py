import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as k
data = pd.read_csv('../input/fer2013/fer2013.csv')
data.head()
num_classes = 7
img_width = 48
img_height = 48
data.shape
data.Usage.value_counts()
X = data['pixels']
y = data['emotion']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
X_train_new = []
for i in X_train:
    X_train_new.append([int(j) for j in i.split()])
X_train_new = np.array(X_train_new)/255.0
X_test_new = []
for i in X_test:
    X_test_new.append([int(j) for j in i.split()])
X_test_new = np.array(X_test_new)/255.0
X_train_new = X_train_new.reshape(X_train_new.shape[0], img_width, img_height, 1)
X_test_new = X_test_new.reshape(X_test_new.shape[0], img_width, img_height, 1)
X_test_new.shape
X_train_new = X_train_new.astype('float32')
X_test_new = X_test_new.astype('float32')
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten,Dropout, MaxPooling2D, Activation, BatchNormalization
from tensorflow.keras.preprocessing import image
model = Sequential()
model.add(Conv2D(filters=64,kernel_size=(5,5),input_shape=(X_train_new.shape[1:]), padding='same'))
model.add(Conv2D(64, kernel_size=(5,5), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(5,5), padding='same'))
model.add(Conv2D(128,kernel_size=(5,5), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same'))
model.add(Conv2D(256,kernel_size=(3,3), padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(7, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
from keras import backend as b
b.set_value(model.optimizer.learning_rate, 0.0001)
history = model.fit(X_train_new, y_train, epochs=20, batch_size=64, validation_data=(X_test_new,y_test))
pd.DataFrame(history.history).tail()
y_pred = model.predict_classes(X_test_new)
from sklearn.metrics import confusion_matrix
cfm = confusion_matrix(np.argmax(y_test, axis=1),y_pred)
cfm