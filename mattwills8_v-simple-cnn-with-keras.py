import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

import matplotlib.pyplot as plt
%matplotlib inline

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
x_train_df = train_df.drop(labels = ["label"],axis = 1) 
y_train_df = train_df['label']

x_train, x_test, y_train, y_test = train_test_split(x_train_df, y_train_df)
print(x_train.shape)
print(x_test.shape)
# 28x28 = 784
img_rows, img_cols = 28, 28
batch_size = 32
num_classes = 10
epochs = 20
x_train = np.reshape(x_train.values, (x_train.shape[0], img_rows, img_cols, 1))
x_test = np.reshape(x_test.values, (x_test.shape[0], img_rows, img_cols, 1))
pred = np.reshape(test_df.values, (test_df.shape[0], img_rows, img_cols, 1))

input_shape = (img_rows, img_cols, 1)
x_train.shape
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
def printer(model,img):
    img_batch = np.expand_dims(img, axis=0)
    print(img_batch.shape)
    img_pred = model.predict(img_batch)
    print(img_pred.shape)
    
    img_pred = np.squeeze(img_pred, axis=0)
    print(img_pred.shape)
    img_pred = img_pred.reshape((32,13,13))
    print(img_pred.shape)
    plt.imshow(img_pred[5])
model = Sequential()

model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                 input_shape=input_shape))
model.add(BatchNormalization(axis=2, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
img = x_train[8]
printer(model, img)

plt.imshow(img.reshape(img.shape[:2]))
'''model = Sequential()

model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                 input_shape=input_shape))
model.add(BatchNormalization(axis=2, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',
                 input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(7, 7),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(5, 5),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=(2, 2),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

annealer = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
validation_data=(x_test, y_test))'''
'''results = model.predict(pred)
results = np.argmax(results, axis=1)
results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("cnn_mnist_datagen.csv",index=False)'''
