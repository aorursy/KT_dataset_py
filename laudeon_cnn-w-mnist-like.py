# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data_test = pd.read_csv("../input/fashion-mnist_test.csv")

data_train = pd.read_csv("../input/fashion-mnist_train.csv")



# print(data_train.shape)
import keras

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



# Pre process train data

X_train = np.array(data_train.iloc[:, 1:])

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

y_train = to_categorical(np.array(data_train.iloc[:, 0]))



# Pre process test data

X_test = np.array(data_test.iloc[:, 1:])

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_test = to_categorical(np.array(data_test.iloc[:, 0]))



# Optimizing

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255
batch_size = 128

num_classes = 10

epochs = 15



model = Sequential()



model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation="relu"))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(64, (3, 3), activation="relu"))

model.add(MaxPooling2D((2, 2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(num_classes, activation="softmax"))



optimizer = keras.optimizers.Adam()

model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
print(model.summary())



history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

score = model.evaluate(X_test, y_test)



print('Test loss:', score[0])

print('Test accuracy:', score[1])