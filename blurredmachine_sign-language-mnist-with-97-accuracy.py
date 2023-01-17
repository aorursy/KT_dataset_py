import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
mnist_train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train.csv")
mnist_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test.csv")
print(mnist_train.shape, mnist_test.shape)
mnist_train.head()
mnist_test.head()
print(mnist_train.isna().any().any(), mnist_test.isna().any().any())
# Data is completely clean without any missing values.
mnist_train_data = mnist_train.loc[:, "pixel1":]
mnist_train_label = mnist_train.loc[:, "label"]

mnist_test_data = mnist_test.loc[:, "pixel1":]
mnist_test_label = mnist_test.loc[:, "label"]
# Data Normalization
mnist_train_data = mnist_train_data/255.0
mnist_test_data = mnist_test_data/255.0
data_array = np.array(mnist_train_data.loc[2, :])
shaped_data = np.reshape(data_array, (28, 28))
sign_img = plt.imshow(shaped_data, cmap=plt.cm.binary)
plt.colorbar(sign_img)
print("IMAGE LABEL: {}".format(mnist_train.loc[2, "label"]))
plt.show()
sns.countplot(mnist_train.label)
print(list(mnist_train.label.value_counts().sort_index()))
mnist_train_data = np.array(mnist_train_data)
mnist_test_data = np.array(mnist_test_data)

mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 28, 28, 1)
mnist_test_data = mnist_test_data.reshape(mnist_test_data.shape[0], 28, 28, 1)

print(mnist_train_data.shape, mnist_train_label.shape)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D, AvgPool2D
from tensorflow.keras.optimizers import Adadelta
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler
nclasses = mnist_train_label.max() - mnist_train_label.min() + 1
mnist_train_label = to_categorical(mnist_train_label, num_classes = nclasses)
print("Shape of ytrain after encoding: ", mnist_train_label.shape)
nclasses = mnist_test_label.max() - mnist_test_label.min() + 1
mnist_test_label = to_categorical(mnist_test_label, num_classes = nclasses)
print("Shape of ytest after encoding: ", mnist_test_label.shape)
model = Sequential()
model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (28, 28, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 3, activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(128, kernel_size = 4, activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(25, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model_history = model.fit(mnist_train_data, mnist_train_label, batch_size=500, shuffle=True, epochs=20, validation_split=0.1)
mnist_train_data = mnist_train_data.reshape(mnist_train_data.shape[0], 784)
print(mnist_train_data.shape, mnist_train_label.shape)

vc_loss, vc_accuracy = model.evaluate(mnist_test_data, mnist_test_label)
print("\nLOSS: {}\nACCURACY: {}".format(vc_loss, vc_accuracy))
plt.plot(model_history.history['accuracy'],label = 'ACCURACY')
plt.plot(model_history.history['val_accuracy'],label = 'VALIDATION ACCURACY')
plt.legend()
plt.plot(model_history.history['loss'],label = 'TRAINING LOSS')
plt.plot(model_history.history['val_loss'],label = 'VALIDATION LOSS')
plt.legend()