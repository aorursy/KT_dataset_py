# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
train = pd.read_csv("../input/emnist-bymerge-train.csv")
test = pd.read_csv("../input/emnist-bymerge-test.csv")
def load_char_mappings(mapping_path):
    """
    load EMNIST character mappings. This maps a label to the correspondent byte value of the given character
    return: the dictionary of label mappings
    """
    mappings = {}
    with open(mapping_path) as f:
        for line in f:
            (key, val) = line.split()
            mappings[int(key)] = int(val)

    return mappings
def rotate(img):
    flipped = np.fliplr(img.reshape(28,28))
    return np.rot90(flipped).reshape(784,)
        
for i in range(len(x_test)):
    x_test[i] = rotate(x_test[i])
def rotate(img):
    flipped = np.fliplr(img.reshape(28,28))
    return np.rot90(flipped).reshape(784,)
        
for i in range(len(x_test)):
    x_train[i] = rotate(x_train[i])
num_classes = 47
y_train = train.iloc[:,0]
y_train = np_utils.to_categorical(y_train, num_classes)
print ("y_train:", y_train.shape)
x_train = train.iloc[:,1:]
x_train = x_train.astype('float32')
x_train /= 255
print ("x_train:",x_train.shape)
y_test = test.iloc[:,0]
y_test = np_utils.to_categorical(y_test, num_classes)
print ("y_test:", y_test.shape)
x_test = test.iloc[:,1:]
x_test = x_test.astype('float32')
x_test /= 255
print ("x_test:",x_test.shape)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
print(x_train.shape)
print(x_test.shape)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200, verbose=2)
model.save('my_model.h5')
from subprocess import check_output
print(check_output(["ls", "../working/"]).decode("utf8"))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
c = x_test[89613]
c = c.reshape(28,28)
plt.gray()
plt.imshow(c.reshape(28,28))
