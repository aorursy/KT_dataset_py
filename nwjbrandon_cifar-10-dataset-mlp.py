# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from keras.utils import to_categorical
from keras.datasets import *
import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import *
from keras.models import Sequential, load_model
from keras.layers import *
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
import os
import shutil

np.random.seed(24601)

def plot_history(history):
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
def load_cifar10():
    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    train_x = train_x.reshape(train_x.shape[0], 3072)
    test_x = test_x.reshape(test_x.shape[0], 3072) 
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x /= 255.0
    test_x /= 255.0
    ret_train_y = to_categorical(train_y,10)
    ret_test_y = to_categorical(test_y, 10)
    
    return (train_x, ret_train_y), (test_x, ret_test_y)


(train_x, train_y), (test_x, test_y) = load_cifar10()
def buildmodel(model_name):
    # Create the neural network
    model = Sequential()
    model.add(Dense(2048, input_shape = (3072, ), activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(1024, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(512,activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(10, activation = 'softmax'))

    sgd = SGD(lr = 0.1, momentum=0.01)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = 'accuracy')
    return model
def train(model, train_x, train_y, epochs, test_x, test_y, model_name):
    savemodel = ModelCheckpoint(model_name)
    stopmodel = EarlyStopping(min_delta=0.001, patience=10)
    
    print("Starting training.")
    history = model.fit(x=train_x, y=train_y, batch_size=32, validation_data=(test_x, test_y), verbose=True, epochs=epochs, callbacks=[savemodel, stopmodel])
    plot_history(history)
    print("Done. Now evaluating.")
    
    loss, acc = model.evaluate(x=test_x, y=test_y)
    print("Test accuracy: %3.2f, loss: %3.2f"%(acc, loss))
MODEL_NAME = 'cifar.hd5'
(train_x, train_y), (test_x, test_y) = load_cifar10()
model = buildmodel(MODEL_NAME)
train(model, train_x, train_y, 100, test_x, test_y, MODEL_NAME)