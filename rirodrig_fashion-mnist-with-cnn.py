# load basic libraries 
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt 
# train-test split using sklearn
from sklearn.model_selection import train_test_split 
import keras 
#Convert (integer) target to categorical values (vector)
from keras.utils import to_categorical 
#CNN Model building libraries 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten 
from keras.layers.advanced_activations import LeakyReLU

train_data = pd.read_csv("../input/fashion-mnist_train.csv")
print(train_data.head(1))

test_data = pd.read_csv("../input/fashion-mnist_test.csv")
print(test_data.head(1))
# checking for any null values
print(train_data.isnull().sum().sum())
print(test_data.isnull().sum().sum())
# train data
X = train_data.drop(['label'], axis=1).values
y = to_categorical(train_data['label'].values)
#Split train data 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)
# test data
X_test = test_data.drop(['label'], axis=1).values
y_test = to_categorical(test_data['label'].values)
#Reshape data 
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

# reshape the data
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_val = X_val.astype('float32')
X_train /= 255
X_test /= 255
X_val /= 255

print("Train Data shape")
print(X_train.shape)
print(y_train.shape)
print("Validation Data Shape")
print(X_val.shape)
print(y_val.shape)
plt.imshow(X_train[1, :].reshape((28,28)))
#input image dimensions

batch_size = 64
num_classes = 10
epochs = 50

fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 kernel_initializer='he_normal',
                 input_shape=input_shape))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2, 2)))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64, (3, 3), activation='relu'))
fashion_model.add(MaxPooling2D(pool_size=(2, 2)))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3, 3), activation='relu'))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation='relu'))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes, activation='softmax'))



fashion_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
fashion_model.summary()
fashion_train = fashion_model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_val, y_val))
test_eval = fashion_model.evaluate(X_test, y_test, verbose=0)

test_eval
# Accuracy results
accuracy = fashion_train.history['acc']
val_accuracy = fashion_train.history['val_acc']
loss = fashion_train.history['loss']
val_loss = fashion_train.history['val_loss']
epochs = range(len(accuracy))
#  Plots
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()