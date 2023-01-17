import numpy as np

import pandas as pd

import os



import matplotlib.pyplot as plt

import cv2
train=pd.read_csv('../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

test=pd.read_csv('../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
train.head()
test.head()
len(train)/10
def get_data(df):

    X=[]

    y=[]

    for arr in df.values:

        img=arr[1:].reshape(28,28)

        X.append(img)

        y.append(arr[0])

    return X,y
xtrain,ytrain = get_data(train)

xtest,ytest = get_data(test)
xtrain=(np.array(xtrain)/255).reshape(-1,28,28,1)

xtest=(np.array(xtest)/255).reshape(-1,28,28,1)



from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

ytrain = ohe.fit_transform(np.array(ytrain).reshape(-1,1)).todense()

ytest = ohe.fit_transform(np.array(ytest).reshape(-1,1)).todense()

import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
input_shape_=(28,28,1)



model = Sequential()



model.add(Conv2D(filters=64, kernel_size=(3,3), input_shape=input_shape_, activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.2))



model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Dropout(0.2))



model.add(Flatten())



model.add(Dense(128, activation='relu'))



model.add(Dense(64, activation='relu'))



model.add(Dense(24, activation='softmax'))



model.summary()
model.compile(optimizer='adam',

              loss='categorical_crossentropy',

              metrics=['accuracy'])



history = model.fit(xtrain, ytrain, validation_split=0.2, epochs=10)
# Accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Training Accuracy vs Validation Accuracy')

plt.show()



# Loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Training Loss vs Validation Loss')

plt.show()

model.evaluate(xtest,ytest)