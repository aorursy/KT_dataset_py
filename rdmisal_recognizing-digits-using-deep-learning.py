import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras 

from keras.preprocessing.image import img_to_array, array_to_img
data=pd.read_csv('../input/train.csv')

x_train=data.drop('label',axis=1)

y_train=data[['label']]

test=pd.read_csv('../input/sample_submission.csv')

y_test=test[['Label']]

x_test=pd.read_csv('../input/test.csv')
image=x_train[:1]

image1=image.as_matrix()

image2=image1.reshape(28,28)

plt.figure(28, figsize=(10,10))

plt.imshow(image2, interpolation='nearest', cmap=plt.cm.gray_r)

plt.show()
image=x_train[3:4]

image1=image.as_matrix()

image2=image1.reshape(28,28)

plt.figure(28, figsize=(10,10))

plt.imshow(image2, interpolation='nearest', cmap=plt.cm.gray_r)

plt.show()
import keras

from keras.utils import to_categorical

num_classes=10

y_train = keras.utils.to_categorical(y_train, num_classes)

y_test=keras.utils.to_categorical(y_test, num_classes)
def Model():

    from keras.models import Sequential

    from keras.layers import  Dense, Dropout

    model=Sequential()

    model.add(Dense(50, activation='relu', input_shape=(784,)))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    model.summary()

    return (model)
model=Model()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from  keras.callbacks import EarlyStopping

early_stoping_moniter=EarlyStopping(patience=2)

batch_size=500

epochs=20

predict=model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,callbacks=[early_stoping_moniter])
predict=model.predict(x_test)
test1=np.array(x_test)

plt.imshow(test1[500].reshape(28,28))
from numpy import argmax

prediction = [np.argmax(y, axis=None, out=None) for y in predict]

prediction[500]
submission = pd.DataFrame({

        "ImageId": test["ImageId"],

        "Label": prediction

    })

submission.to_csv('submission.csv', index=False)