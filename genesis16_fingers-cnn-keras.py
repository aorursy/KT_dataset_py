import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import io, transform

import os, glob

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

from sklearn.model_selection import train_test_split
print("Loaded...")
train_img_list = glob.glob("../input/fingers/fingers/train/*.png")
test_img_list = glob.glob("../input/fingers/fingers/test/*.png")
print(len(train_img_list),
     len(test_img_list), sep = '\n')
def import_data():
    train_img_data = []
    test_img_data = []
    train_label_data = []
    test_label_data = []
    
    for img in train_img_list:
        img_read = io.imread(img, channels = 1)
        img_read = transform.resize(img_read, (128,128), mode = 'constant')
        train_img_data.append(img_read)
        train_label_data.append(img[-5])
    
    for img in test_img_list:
        img_read = io.imread(img, channels = 1)
        img_read = transform.resize(img_read, (128,128), mode = 'constant')
        test_img_data.append(img_read)
        test_label_data.append(img[-5])
        
    return np.array(train_img_data), np.array(test_img_data), np.array(train_label_data), np.array(test_label_data)
    
xtrain, xtest, ytrain, ytest = import_data()
xtrain = xtrain.reshape(xtrain.shape[0], 128, 128, 1)
xtest = xtest.reshape(xtest.shape[0], 128, 128, 1)

ytrain = tf.keras.utils.to_categorical(ytrain, num_classes = 6)
ytest = tf.keras.utils.to_categorical(ytest, num_classes = 6)
print(xtrain.shape, xtest.shape, ytrain.shape, ytest.shape)
x_train, x_test, y_train, y_test = train_test_split(xtrain, ytrain, test_size = 0.20, random_state = 7, shuffle = True)
x_train_val, x_test_val, y_train_val, y_test_val = train_test_split(x_train, y_train, test_size = 0.20, random_state = 7, shuffle = True)
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (128, 128, 1), activation = 'relu'))
model.add(Conv2D(32, (3,3), activation = 'relu'))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(Conv2D(64, (3,3), activation = 'relu'))

model.add(MaxPool2D((2,2)))

model.add(Conv2D(128, (3,3), activation = 'relu'))
model.add(Conv2D(128, (3,3), activation = 'relu'))

model.add(Flatten())

model.add(Dropout(0.40))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.40))
model.add(Dense(6, activation = 'softmax'))

model.summary()
model.compile('SGD', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = x_train, y = y_train, batch_size = 128, epochs = 10, validation_data = (x_test, y_test))
pred = model.evaluate(xtest,
                      ytest,
                    batch_size = 128)

print("Accuracy of model on test data is: ",pred[1]*100)
