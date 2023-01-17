from tensorflow import keras

keras.__version__
import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

%matplotlib inline

from subprocess import check_output
df_train_file = "../input/fashion-mnist_train.csv"

df_test_file = "../input/fashion-mnist_test.csv"
df_train = pd.read_csv(df_train_file)

df_test = pd.read_csv(df_test_file)

df_train.head()
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



X = np.array(df_train.iloc[:, 1:])

y = to_categorical(np.array(df_train.iloc[:, 0]))



#Here we split validation data to optimiza classifier during training

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=13)



#Test data

X_test = np.array(df_test.iloc[:, 1:])

y_test = to_categorical(np.array(df_test.iloc[:, 0]))







X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 1)



X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_val = X_val.astype('float32')

X_train /= 255

X_test /= 255

X_val /= 255
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.layers.normalization import BatchNormalization



batch_size = 128

num_classes = 10

epochs = 2



#input image dimensions

img_rows, img_cols = 28, 28



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),

                 activation='relu',

                 kernel_initializer='he_normal',

                 input_shape=input_shape))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))

model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adam(),

              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(X_val, y_val))

score = model.evaluate(X_test, y_test, verbose=0)
def get_features_label(df):

    features = df.values[:,1:]/255

    

    labels = df['label'].values

    return features, labels
train_features, train_labels = get_features_label(df_train)

test_features, test_labels = get_features_label(df_test)
print(train_features.shape)

print(test_features.shape)
#take a peak at some values in an image

train_features[20, 300:320]
example_index = 221

plt.figure()

_ = plt.imshow(np.reshape(train_features[example_index, :],(28,28)), 'gray')
train_labels.shape
train_labels[example_index]
train_labels = tf.keras.utils.to_categorical(train_labels)

test_labels = tf.keras.utils.to_categorical(test_labels)
train_labels.shape
train_labels[example_index]