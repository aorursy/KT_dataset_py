import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train_path = os.path.join("../input", "train.csv")
digits = pd.read_csv(train_path)
X = digits.drop('label', axis=1)
y = digits['label'].copy()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from scipy.ndimage.interpolation import shift, rotate, zoom
from sklearn.utils import shuffle

def shift_image(X_in, vertical, horizonal):
    shifted_images = np.empty([len(X_in.values), 784])
    for i, image in enumerate(X_in.values):
        img = shift(image.reshape(28, 28), [vertical, horizonal], cval=0)
        shifted_images[i] = img.ravel()
    return pd.DataFrame(data=shifted_images, columns=X_in.columns)

def rotate_image(X_in, angle):
    shifted_images = np.empty([len(X_in.values), 784])
    for i, image in enumerate(X_in.values):
        img = rotate(image.reshape(28, 28), angle, cval=0, reshape=False)
        shifted_images[i] = img.ravel()
    return pd.DataFrame(data=shifted_images, columns=X_in.columns)

def zoom_image(X_in):
    shifted_images = np.empty([len(X_in.values), 784])
    for i, image in enumerate(X_in.values):
        img = zoom(image.reshape(28, 28), 1.3)
        img = img[5:33, 5:33]
        shifted_images[i] = img.ravel()
    return pd.DataFrame(data=shifted_images, columns=X_in.columns)

X_train_shift = X_train.append(rotate_image(X_train, 10))
X_train_shift = X_train_shift.append(rotate_image(X_train, -10))
X_train_shift = X_train_shift.append(shift_image(X_train, 5, 0))
X_train_shift = X_train_shift.append(shift_image(X_train, -5, 0))
X_train_shift = X_train_shift.append(shift_image(X_train, 0, -5))
X_train_shift = X_train_shift.append(shift_image(X_train, 0, 5))
X_train_shift = X_train_shift.append(zoom_image(X_train))

y_train_shift = y_train.append(y_train)
y_train_shift = y_train_shift.append(y_train)
y_train_shift = y_train_shift.append(y_train)
y_train_shift = y_train_shift.append(y_train)
y_train_shift = y_train_shift.append(y_train)
y_train_shift = y_train_shift.append(y_train)
y_train_shift = y_train_shift.append(y_train)

X_train_shift, y_train_shift = shuffle(X_train_shift, y_train_shift, random_state=0)
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train_shift)
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as backend

img_rows, img_cols = 28, 28
batch_size = 128
n_classes = len(np.unique(y_train))

# transform train data and targets to keras format.
if backend.image_data_format() == 'channels_first':
    X_train_reshape = X_train_scaled.reshape(X_train_scaled.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train_reshape = X_train_scaled.reshape(X_train_scaled.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_train_onehot = keras.utils.to_categorical(y_train_shift, n_classes)

#building CNN
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
#model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dropout(0.5))
#model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=n_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


history = model.fit(X_train_reshape, y_train_onehot, batch_size=batch_size, epochs=20, verbose=2)
y_nn_train_predict = model.predict_classes(X_train_reshape, verbose=0)
total = sum(y_train_shift == y_nn_train_predict)
total/len(X_train_reshape)
X_test_scaled = scaler.transform(X_test)
# transform train data and targets to keras format.
if backend.image_data_format() == 'channels_first':
    X_test_reshape = X_test_scaled.reshape(X_test_scaled.shape[0], 1, img_rows, img_cols)
else:
    X_test_reshape = X_test_scaled.reshape(X_test_scaled.shape[0], img_rows, img_cols, 1)

y_nn_test_predict = model.predict_classes(X_test_reshape, verbose=0)
confusion_matrix(y_test, y_nn_test_predict)
total = sum(y_test == y_nn_test_predict)
total/len(X_test_reshape)
import csv

test_path = os.path.join("../input", "test.csv")
digit_test = pd.read_csv(test_path)
digit_test_scaled = scaler.transform(digit_test)

if backend.image_data_format() == 'channels_first':
    digit_test_reshape = digit_test_scaled.reshape(digit_test_scaled.shape[0], 1, img_rows, img_cols)
else:
    digit_test_reshape = digit_test_scaled.reshape(digit_test_scaled.shape[0], img_rows, img_cols, 1)

digit_predict = model.predict_classes(digit_test_reshape)


try:
    os.remove("./result.csv")
except FileNotFoundError:
    print("result.csv not found")

csvfile = open("result.csv", "w", encoding="utf-8")
writer = csv.writer(csvfile)
writer.writerow(["ImageId", "Label"])

for i, d in enumerate(digit_predict):
    writer.writerow([i+1, d])

csvfile.close()
