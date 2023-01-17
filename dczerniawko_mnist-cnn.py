# data manipulation
import numpy as np
import pandas as pd
import random

# high-level neural networks API - running on top of TensorFlow
import keras
# Sequential is a linear stack of layers
from keras.models import Sequential
# Dense, Flatten - type of layers, Dropout - tool, which decrease chance of overfitting
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K

# data visualisation, live training loss plot
import matplotlib.pyplot as plt

import time
from sklearn.model_selection import train_test_split
# load dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1) 
# shape of data
X_train.shape
def vis_rand_dig():
    # size of pixcel
    plt.figure(figsize = (10, 10))
    # print random digit
    rand_indexes = np.random.randint(0, X_train.shape[0], 16)
    for index,im_index in enumerate(rand_indexes):
        plt.subplot(4, 4, index+1)
        plt.imshow(X_train.values[im_index].reshape(28,28), cmap = 'ocean', interpolation = 'none')
        plt.title('Class %d' % y_train[im_index])
    plt.tight_layout()
vis_rand_dig()
def prep_data(X_train, y_train, test):
   
    X_train = X_train.astype('float32') / 255
    test = test.astype('float32')/255
    X_train = X_train.values.reshape(-1,28,28,1)
    test = test.values.reshape(-1,28,28,1)
    y_train = keras.utils.np_utils.to_categorical(y_train)
    classes = y_train.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = int(time.time()))
    
    return X_train, y_train, X_test, y_test, classes, test
X_train, y_train, X_test, y_test, out_neurons, test = prep_data(X_train, y_train, test)
def cnn():
    cnn = Sequential([
        Conv2D(32, kernel_size = (3, 3), padding = 'same', activation = 'relu', input_shape = (28,28,1)),
        Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (2, 2)),
        Dropout(0.25),
        
        Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (2, 2)),
        Dropout(0.25),
        
        Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same'),
        MaxPool2D(pool_size = (2, 2)),
        Dropout(0.25),
        
        Flatten(),
        
        Dense(512, activation = 'relu'),
        Dropout(0.5),
        Dense(256, activation = 'relu'),
        Dropout(0.5),
        Dense(out_neurons, activation = 'softmax')
    ])
    return cnn

model = cnn()
model.summary()
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam' , metrics = ['accuracy'])
model.fit(X_train, y_train,
          batch_size = 512,
          epochs = 11,
          validation_data = (X_test, y_test),
          verbose = 0);
result = model.evaluate(X_test, y_test, verbose = 0)
print('Accuracy: ', result[1])
print('Error: %.2f%%' % (100- result[1]*100))
y_pred = model.predict(test, verbose=0)
def error_predict(y_test, y_pred):
    for idx, (a, b) in enumerate(zip(y_test, y_pred)):
        if np.argmax(a) == np.argmax(b): continue
        yield idx, np.argmax(a), tuple(np.argsort(b)[-2:])
def display_errors():
    random_errors = random.sample(list(error_predict(y_test, y_pred)), 12)

    plt.figure(figsize=(10, 10))
    X_test_plot = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2] )

    for index, (im_index, y_test_val, (y_pred_2, y_pred_1)) in enumerate(random_errors):
            plt.subplot(4,4,index+1)
            plt.imshow(X_test_plot[im_index], cmap='ocean', interpolation='none')
            plt.title('True value: {0}\nFirst predicted: {1}\nSecond predicted: {2}'.format(y_test_val, y_pred_1, y_pred_2))
            plt.tight_layout()
display_errors()
solution = np.argmax(y_pred,axis = 1)
solution = pd.Series(solution, name="Label").astype(int)
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),solution],axis = 1)
submission.to_csv("mnist_cnn.csv",index=False)