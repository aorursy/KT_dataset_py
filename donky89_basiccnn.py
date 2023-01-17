# Import of needed libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
!ls ../input/fashionmnist/
# Read the dataset with pandas
train_df = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')
test_df  = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
# Get the train and the test in numpy format
x_train, y_train, x_test, y_test = train_df.drop(['label'], axis=1).to_numpy(), train_df['label'].to_numpy(), test_df.drop(['label'], axis=1).to_numpy(), test_df['label'].to_numpy() 

# We need to reshape in order to have a grid of pixels
x_train = x_train.reshape(x_train.shape[0], 28, -1)
x_test = x_test.reshape(x_test.shape[0], 28, -1)
x_train[1].shape
# Print a random image
import matplotlib.pyplot as plt
from numpy.random import randint
plt.imshow(x_train[randint(len(x_train))])
plt.show()

# The required dimension for Convolution is (28, 28, 1), so we have missing channel and we need to add a fake dimension
x_train, x_test = np.expand_dims(x_train, axis=-1), np.expand_dims(x_test, axis=-1)
print(x_train.shape)
# Let's build the model with functional API
n_classes = len(set(y_train))
i = Input(shape=x_train[0].shape)
x = Conv2D(64, kernel_size=(3, 3), activation='relu', strides=2)(i)
x = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=2)(x)
x = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=2)(x)
x = Flatten()(x)
x= Dense(128, activation='relu')(x)
x= Dense(n_classes, activation='softmax')(x)

# Compile and show the model
model = Model(i, x)
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.summary()
# Now train the model
h = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
plt.plot(pd.DataFrame(h.history))
plt.legend(h.history.keys())
plt.show()
i = Input(shape=x_train[0].shape)
x = Conv2D(64, kernel_size=(3, 3), activation='relu')(i)
x = MaxPooling2D(padding='same')(x)
x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(padding='same')(x)
x = Conv2D(16, kernel_size=(3, 3), activation='relu')(x)
x = MaxPooling2D(padding='same')(x)
x = Flatten()(x)
x= Dense(128, activation='relu')(x)
x= Dense(n_classes, activation='softmax')(x)

# Compile and show the model
model = Model(i, x)
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
model.summary()
# Now train the model
h = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
plt.plot(pd.DataFrame(h.history))
plt.legend(h.history.keys())
plt.show()
