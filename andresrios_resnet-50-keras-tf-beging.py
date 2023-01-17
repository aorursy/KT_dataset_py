# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from os import *
import os, cv2, random
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
import numpy as np
from keras.datasets import mnist

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
%matplotlib inline 

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

from keras import backend as K
model = ResNet50(weights='imagenet')
from keras.applications.vgg16 import VGG16
model_vgg16 = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
#model_vgg16.summary()
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    img =cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(file, cv2.COLOR_BGR2RGB)

    return np.array(img).reshape((3,224,224))


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS,CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data
TRAIN_DIR = "../input/dogs-vs-cats-redux-kernels-edition/train/"
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'
ROWS = 224
COLS = 224
CHANNELS = 3
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog.' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat.' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]

# slice datasets for memory efficiency on Kaggle Kernels, delete if using full dataset
train_images = train_dogs[:10000] + train_cats[:10000]
random.shuffle(train_images)
test_images =  test_images[:25] + train_dogs[-150:] + train_cats[-150:]


len(test_images)
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    img =cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return np.array(img).reshape((3,224,224))


def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLS,CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.T
        if i%500 == 0: print('Processed {} of {}'.format(i, count))
    
    return data


train = prep_data(train_images)
test = prep_data(test_images)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))
labels = []
for i in train_images:
    if 'dog.' in i:
        labels.append(1)
    else:
        labels.append(0)

sns.countplot(labels)
plt.xticks(np.arange(2),['Dogs','Cats'])
plt.title('Cats and Dogs')
from keras.utils import to_categorical
labels = to_categorical(labels)
def show_cats_and_dogs(idx):
    dog = read_image(train_dogs[idx])
    dog = np.array(dog).reshape((224,224,3))

    cat = read_image(train_cats[idx])
    cat = np.array(cat).reshape((224,224,3))

    pair = np.concatenate((cat, dog), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()
    
for idx in range(0,2):
    show_cats_and_dogs(idx)
model.summary()
#Create a New Model based on ResNEt50 
input_shape = (224, 224, 3)

K.set_learning_phase(1)
base_model = ResNet50(weights='imagenet', include_top=True, input_shape=input_shape)
base_model.layers.pop()
base_model.outputs = [base_model.layers[-1].output]
base_model.layers[-1].outbound_nodes = []

x = base_model.layers[-1].output
#x = Flatten(name='flatten')(x)
predictions = Dense(2, activation='softmax', name='predictions')(x)
model_01 = Model(inputs=base_model.input, outputs=predictions)

for layer in model.layers[0:111]:
    layer.trainable = False
optimizer = RMSprop(lr=1e-4)
model_01.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
len(model_01.layers)
#optimizer = RMSprop(lr=1e-4)
#objective = 'binary_crossentropy'
#model_01.compile(loss=objective, optimizer=optimizer, metrics=['accuracy'])

nb_epoch = 15
batch_size = 50

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')        
        
def run_catdog():
    
    history = LossHistory()
    model_01.fit(x=train, y=labels, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])
    

    predictions = model_01.predict(test, verbose=0)
    return predictions, history

predictions, history = run_catdog()
loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()
for i in range(0,10):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))
     
    x=np.array(test[i].T).reshape((224,224,3))
    plt.imshow(x)
    plt.show()
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
model_01.weights
#print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition/test"))
predictions
len(test)
img_path = test_images[18]
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
#x = test[0]

x = np.expand_dims(x, axis=0)
'''x = np.float32(x)
print(x.shape,
x.dtype)'''
x = preprocess_input(x)

preds = model_01.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', (preds.shape))
model.input
def show_cats_and_dogs(idx):
    dog = read_image(test_images[idx])
    dog = np.array(dog).reshape((224,224,3))

    plt.figure(figsize=(10,5))
    plt.imshow(dog)
    plt.show()
    
show_cats_and_dogs(0)
img_path = '../input/tets-resnet/sasha.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])
model_vgg16
#Create a New Model based on VGG 16
input_shape = (224, 224, 3)

K.set_learning_phase(1)
base_model = model_vgg16
base_model.layers.pop()
base_model.outputs = [base_model.layers[-1].output]
base_model.layers[-1].outbound_nodes = []

x = base_model.layers[-1].output
#x = Flatten(name='flatten')(x)
predictions = Dense(2, activation='softmax', name='predictions')(x)
model_02 = Model(inputs=base_model.input, outputs=predictions)

#for layer in model.layers[0:111]:
#    layer.trainable = False
optimizer = RMSprop(lr=1e-4)
model_02.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
nb_epoch = 15
batch_size = 150

## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')        
        
def run_catdog():
    
    history = LossHistory()
    model_02.fit(x=train, y=labels, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])
    

    predictions = model_02.predict(test, verbose=0)
    return predictions, history

predictions, history = run_catdog()
loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()
