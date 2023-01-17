# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import cv2                 
import os                  
from tqdm import tqdm
from random import shuffle

!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip -d train
!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip -d test
TRAIN_DIR = '../working/train/train/'
TEST_DIR = '../working/test/test/'
IMG_SIZE = 64
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
len(train_cats), len(train_dogs)
type(train_cats)
train_cats[1]
#Creating training and validation splits
train_list = train_cats[:10000] + train_dogs[:10000]
val_list = train_cats[10000:] + train_dogs[10000:]
len(train_list), len(val_list)
#Function for defining label
def label_img(img):
    if 'cat' in img: return [0, 1]
    elif 'dog' in img: return [1, 0]
#Function for resizing images
def create_train_data(train_list):
    training_data = []
    for img in tqdm(train_list):
        label = label_img(img)
        path = img
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),label])
    shuffle(training_data)
    #np.save('train_data.npy', training_data)
    return training_data
train_list[0]
#Creating array with data
train = create_train_data(train_list)
val = create_train_data(val_list)
def fullprint(*args, **kwargs):
  from pprint import pprint
  import numpy
  opt = numpy.get_printoptions()
  numpy.set_printoptions(threshold=numpy.inf)
  pprint(*args, **kwargs)
  numpy.set_printoptions(**opt)
fullprint(train[1])
len(train)
train[1000][1]
#Creating training ad Validation arrays
X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE)
Y = [i[1] for i in train]

val_X = np.array([i[0] for i in val]).reshape(-1,IMG_SIZE,IMG_SIZE)
val_Y = [i[1] for i in val]
#Scaling data for neural network
X = X/float(255)
val_X = val_X/float(255)
X.shape
fullprint(X[0])

Y = np.asarray(Y)

val_Y = np.asarray(val_Y)
X.shape, val_X.shape
Y.shape, val_Y.shape
#Importing keras libraries
import keras
from keras.layers import Input, Dense, Flatten, Dropout
from keras.optimizers import SGD, Adam
from keras.models import Model
#Network Architecture
x_input = Input(shape=(IMG_SIZE, IMG_SIZE))
x = Flatten()(x_input)
#x = Dense(4096, activation="relu")(x)
#x = Dropout(0.2)(x)
#x = Dense(8096, activation="relu")(x)
x = Dense(1024, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(64, activation="relu")(x)
x = Dense(16, activation="relu")(x)
x_out = Dense(2, activation="softmax")(x)

#Specifying input and output
model = Model(inputs=x_input, outputs=x_out)
sgd = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=sgd, loss = "binary_crossentropy", metrics=["accuracy"])
# Input size * number of nodes + number of nodes for bias
1024*64*64
model.summary()
history = model.fit(x=X, 
                    y=Y, 
                    validation_data=(val_X, val_Y), 
                    epochs = 80, 
                    batch_size=128, 
                    verbose=1)
print(history.history.keys())
import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()