import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras

from keras import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

from keras.optimizers import Adam

from keras.callbacks import TensorBoard



df_train = pd.read_csv("../input/fashion-mnist_train.csv")

df_test = pd.read_csv("../input/fashion-mnist_test.csv")



print(df_train.head())
train_data = np.array(df_train, dtype='float32')

test_data = np.array(df_test, dtype='float32')



X_train = train_data[:,1:]/255

y_train = train_data[:,0]



X_test = test_data[:,1:]/255

y_test = test_data[:,0]
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

image = X_train[100,:].reshape((28,28))

plt.imshow(image)

plt.show()
im_rows = 28

im_cols = 28

batch_size = 512

im_shape = (im_rows, im_cols, 1)



X_train = X_train.reshape(X_train.shape[0], *im_shape)

X_test = X_test.reshape(X_test.shape[0], *im_shape)



X_validate = X_validate.reshape(X_validate.shape[0], *im_shape)



print(X_train.shape)

print(X_test.shape)
cnn = Sequential([

    Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=im_shape),

    MaxPooling2D(pool_size=2),

    Dropout(0.2),

    Flatten(),

    Dense(32,activation='relu'),

    Dense(10,activation='softmax')

])
cnn.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

cnn.fit(X_train,y_train, batch_size=batch_size, epochs=10, verbose=1, validation_data=(X_validate,y_validate))