
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print("train",train.shape)
print("test",test.shape)

Y_train = train["label"]
Y_train = np_utils.to_categorical(Y_train, 10)

X_train = train.drop(labels = ["label"],axis = 1) 
X_train = X_train.values.astype('float32')
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_train /= 255

X_test = test.values.astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_test /= 255

print("X_train",X_train.shape)
print("Y_train",Y_train.shape)
print("X_test",X_test.shape)
model = Sequential()

model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first'))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
print(model.output_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1)
predictions = model.predict_classes(X_test, verbose=0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)
