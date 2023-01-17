#To check Tensorflow/Keras is using GPU or not
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import numpy as np # linear algebra
import pandas as pd

import os
print(os.listdir("../input"))
Data = pd.read_csv("../input/train.csv")

target = Data['label']
image_data = Data.drop(['label'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(image_data, target)

print('Training data and target sizes: \n{}, {}'.format(X_train.shape,Y_train.shape))
print('Test data and target sizes: \n{}, {}'.format(X_test.shape,Y_test.shape))
from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
batch_size = 128
num_classes = 10
epochs = 35
x_train = X_train.as_matrix()
x_test = X_test.as_matrix()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(Y_train, num_classes)
y_test = keras.utils.to_categorical(Y_test, num_classes)
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid')) #softmax

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
# save model to JSON
model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)

# save weights to HDF5
model.save_weights("models/model.h5")
print("Saved model to disk")

print('Test loss:', score[0])
print('Test accuracy:', score[1])
#from keras.models import model_from_json

# load json and create model
#json_file = open('models/model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("models/model.h5")
#print("Loaded model from disk")

#loaded_model.summary()
countA, countB = [], []

pred = model.predict_classes(x_test)

#print ('Actual\t', 'Predicted')
for i, (char0,char1) in enumerate(zip(Y_test.values, pred), 1):
    count = 0
    count1 = 0
    #print (i, char0,"\t", char1)
    if char0 == char1:
        count = count + 1
        countA.append(count)
    else:
        count1 = count1 + 1
        countB.append(count1)
print ("True Predicted: {0} \nFalse Predicted: {1}".format(sum(countA), sum(countB)))