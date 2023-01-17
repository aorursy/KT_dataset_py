import keras

import numpy as np

import pandas as pd

import cv2

from matplotlib import pyplot as plt

from keras.models import Sequential 

from keras.layers import Conv2D,MaxPooling2D, Dense,Flatten, Dropout

from keras.datasets import mnist 

import matplotlib.pyplot as plt

from keras.utils import np_utils

from keras.optimizers import SGD
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
print(test.head(3))
y_train = train['label'].values





X_train = train.drop(['label'],axis=1)

X_test = test



X_train = np.array(X_train.iloc[:,:])

X_train = np.array([np.reshape(i, (28,28)) for i in X_train])



X_test = np.array(X_test.iloc[:,:])

X_test = np.array([np.reshape(i, (28,28)) for i in X_test])



num_classes = 10

y_train = np.array(y_train).reshape(-1)



y_train = np.eye(num_classes)[y_train]
X_train = X_train.reshape((42000, 28, 28, 1))

X_test = X_test.reshape((28000, 28, 28, 1))
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))



model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Dropout(0.25))

#flatten since too many dimensions, we only want a classification output

model.add(Flatten())

#fully connected to get all relevant data

model.add(Dense(128, activation='relu'))

#one more dropout for convergence' sake :) 

model.add(Dropout(0.5))

#output a softmax to squash the matrix into output probabilities

model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=32)
img = X_test[1000]

test_img = img.reshape((1,28,28,1))

img_class = model.predict_classes(test_img)

prediction = img_class[0]

classname = img_class[0]

print("Class: ",classname)
img = img.reshape((28,28))

plt.imshow(img)

plt.title(classname)

plt.show()
Y_test = model.predict(X_test)

print(Y_test)
Y_test = np.argmax(Y_test,axis = 1)
print(Y_test)
results = pd.Series(Y_test,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("digit.csv",index=False)