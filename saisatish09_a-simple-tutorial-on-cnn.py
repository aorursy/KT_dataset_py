import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

from keras.datasets import mnist

import seaborn as sns

from keras.initializers import RandomNormal

from keras import backend as K

from keras.models import Sequential 

from keras.layers import Dense, Activation 

from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten

from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt

import numpy as np

import time



def plt_dynamic(x, vy, ty, ax, colors=['b']):

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

X_test = test
X_train = X_train / 255.0

X_test = X_test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

X_test  = X_test.values.reshape(-1,28,28,1)
Y_train[100]  # Here the output image is 9. we need to convert it using one-hot encoding.
Y_train = to_categorical(Y_train, num_classes = 10)

Y_train[[100]]
print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d, %d)"%(X_train.shape[1], X_train.shape[2]))

print("Number of training examples :", X_test.shape[0], "and each image is of shape (%d, %d)"%(X_test.shape[1], X_test.shape[2]))
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=42)
model = Sequential()

model.add(Conv2D(32, kernel_size = (5,5), activation = 'relu', input_shape = (28,28,1), padding = 'same'))

model.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model_fit = model.fit(X_train, Y_train, batch_size = 100, epochs = 10, verbose=1, validation_data = (X_val,Y_val))
y_pred  = model.predict_classes(X_test)
y_pred[:5]
results = pd.Series(y_pred,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("simple_cnn4.csv",index=False)