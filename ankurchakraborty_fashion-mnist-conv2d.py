import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
img_rows, img_cols = 28, 28

raw_train = np.array(pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv"), dtype='float32')
raw_test = np.array(pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv"), dtype='float32')

x_train = raw_train[:, 1:]/255.0
y_train = raw_train[:,0]

x_test = raw_test[:, 1:]/255.0
y_test = raw_test[:,0]

x_train = x_train.reshape(y_train.shape[0], img_rows, img_cols,1)
x_test = x_test.reshape(y_test.shape[0], img_rows, img_cols,1)

print(x_train.shape)
print(x_test.shape)

K = len(set(y_train))
print (K)

plt.imshow(x_train[4500].reshape(28,28))
i = Input(shape=x_test[0].shape)

x = Conv2D(32, (3,3), activation='relu', padding='same') (i)
x = BatchNormalization() (x)
x = Conv2D(32, (3,3), activation='relu', padding='same') (x)
x = BatchNormalization() (x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(64, (3,3), activation='relu', padding='same') (i)
x = BatchNormalization() (x)
x = Conv2D(64, (3,3), activation='relu', padding='same') (x)
x = BatchNormalization() (x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation='relu', padding='same') (i)
x = BatchNormalization() (x)
x = Conv2D(128, (3,3), activation='relu', padding='same') (x)
x = BatchNormalization() (x)
x = MaxPooling2D((2,2))(x)

x = Flatten() (x)

x = Dense(1024, activation='relu') (x)
x = Dropout(0.2) (x)
x = Dense(128, activation = 'relu') (x)
x = Dropout(0.2) (x)
x = Dense(K, activation='softmax') (x)

model = Model(i,x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = 20)
plt.plot(r.history['loss'], label="loss")
plt.plot(r.history['val_loss'], label="val_loss")

plt.plot(r.history['accuracy'], label="accuracy")
plt.plot(r.history['val_accuracy'], label="val_accuracy")

plt.legend()
batch_size = 32
batch_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
train_generator = batch_generator.flow(x_train, y_train, batch_size)
r = model.fit_generator(train_generator, validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0]//batch_size, epochs=20)
plt.plot(r.history['loss'], label="loss")
plt.plot(r.history['val_loss'], label="val_loss")

plt.plot(r.history['accuracy'], label="accuracy")
plt.plot(r.history['val_accuracy'], label="val_accuracy")

plt.legend()