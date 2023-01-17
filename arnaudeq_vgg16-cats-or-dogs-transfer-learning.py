from __future__ import division,print_function
import math, os, json, sys, re
import _pickle as pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain
from imp import reload

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
from IPython.lib.display import FileLink
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import *
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import applications
from scipy.misc import imread, imresize, imsave
from string import Template
from bs4 import BeautifulSoup

def load_array(fname):
    return bcolz.open(fname)[:]


def extract_img_info(xml):
    img_info = dict()
    img_info["size"] = [int(n) for n in xml.size.stripped_strings][:-1]
    img_info["filename"] = xml.find_all("filename")[0].text
    bboxes = dict()
    tmp = dict()
    for obj in xml.find_all("object"):
        img_info["tag"] = obj.find_all("name")[0].text
        for n in obj.find_all("bndbox"):
            tmp["xmin"] = int(n.find_all("xmin")[0].text)
            tmp["ymin"] = int(n.find_all("ymin")[0].text)
            tmp["width"] = int(n.find_all("xmax")[0].text) - tmp['xmin']
            tmp["height"] = int(n.find_all("ymax")[0].text) - tmp['ymin']
    img_info["bboxes"] = tmp
    return img_info

# below you'll need to provide the path of the folder where all the files are listed
def get_list_annotations(path):
    listing = sorted(os.listdir(path))
    annotations = []
    for file in listing:
        xml = open(path + file,'r')
        xml = BeautifulSoup(xml, "lxml")
        img_info = extract_img_info(xml)
        annotations.append(img_info)
    return(annotations)

def dict_to_pd(file):
    d = file
    df = pd.DataFrame([],columns=['filename','img_width','img_height','bb_x','bb_y','bb_width','bb_height'])
    for i in range(0,len(d)):
        to_add = pd.DataFrame([[d[i]['filename'],
                       d[i]['size'][0], 
                       d[i]['size'][1],
                       d[i]['bboxes']['xmin'],
                       d[i]['bboxes']['ymin'],
                       d[i]['bboxes']['width'], 
                       d[i]['bboxes']['height']]], 
                      columns=['filename','img_width','img_height','bb_x','bb_y','bb_width','bb_height'])
        to_add.set_index([[i]],inplace=True)
        df = df.append(to_add)
    return(df)

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)

def get_data(path, target_size=(224,224)):
    batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)
    return np.concatenate([batches.next() for i in range(batches.samples)])
import os
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
print(os.listdir("../input"))
DATA_DIR = '../input/cats-vs-dogs-1000/dogs_cats_sample_1000/dogs_cats_sample_1000/'
DATA_DIR = '../input/cats-vs-dogs-5000/dogs_cats_sample_5000/dogs_cats_sample_5000/'
# defining the dimensions of the images
input_shape =  (224, 224)

# defining the batch-size of the parameters
batch_size = 64

train_data_dir = DATA_DIR + 'train'
validation_data_dir = DATA_DIR + 'valid'
from imageio import imread
from skimage.transform import resize
def plot_examples(path):
    print('Images belonging to class:', path)
    listdir = os.listdir(path)
    fig, ax = plt.subplots(nrows=2,ncols=5,figsize=(20,10))
    ax = ax.ravel()
    for idx, e in enumerate(listdir[:10]):
        img = imread(os.path.join(path, e))
        img = resize(img,input_shape)
        ax[idx].imshow(img)
plot_examples(DATA_DIR + 'train/cats')
plot_examples(DATA_DIR + 'train/dogs')
train_gen = get_batches(dirname=train_data_dir,
                        class_mode='binary',
                        batch_size=1,
                        shuffle=False)
datagen = ImageDataGenerator()

datagen_augmented = ImageDataGenerator(
        shear_range=.2,       # randomly applies shearing transformation
        zoom_range=0.2,        # randomly applies shearing transformation
        rotation_range=20,     # ranndomly rotates images
        width_shift_range=0.2, # randomly shifts the image (width wise)
        height_shift_range=0.2,# randomly shifts the image (height wise)
        horizontal_flip=True,  # randomly flip the images (horizentally) 
        vertical_flip=False      # randomly flip the images (vertically)
)    
img, label = train_gen.next()
datagen_augmented.fit(img)
i=0
for img_batch in datagen_augmented.flow(img, batch_size=9):
    for img_ in img_batch:
        plt.figure()
        img_ = imresize(img_,input_shape)
        plt.imshow(img_)
        i=i+1    
    if i >= 9:
        break
def get_batches(generator, path, class_mode='categorical', batch_size=32, input_shape=(224,224),shuffle=False):
    '''Getting batches by flowing from directory
    Parameters:
    ---------------
    generator: Keras object
        The Image Data Generator Correctly filled
        
    path: str
        The path to train, val or test directory containing at least one class.
        Reminder: classes are organized in sub-folders
    
    class_mode: str, optional (default='categorical')
        Indicate the class mode of the data generator, if 'binary' or 'categorical'
    
    batch_size: int, optional (default=32)
        The size of each batch
    
    input_shape: tuple, optional (default=(224, 224))
        The size  to which the inputs need to be sized.
        Resizing done automatically if necessary.
    
    shuffle: boolean, optional (default=False)
        Wether to shuffle or not when getting batches
        
    Output:
    ---------------
    batches: Keras object
        The batches of images found by the ImageGenerator
    '''
    batches = generator.flow_from_directory(path,
                                            target_size=input_shape,
                                            batch_size=batch_size,
                                            class_mode=class_mode,
                                            shuffle=shuffle)
    return(batches)
# applying Data-Augmentation for train images
train_gen = get_batches(datagen,
                        train_data_dir,
                        class_mode='binary',
                        batch_size=batch_size,
                        shuffle=False)

# Simply Rescaling validation images
valid_gen = get_batches(datagen,
                        validation_data_dir,
                        class_mode='binary',
                        batch_size=batch_size,
                        shuffle=False)
def get_VGG16(trainable=False,
              pop=True):
    '''It calls the convolutional part of the vgg model. 
    The model will mainly serve as feature extractor from the images
    Parameters:
    -------------
    trainable: Boolean, optional (default=False) 
        If to train the convolutional layers or not
        
    pop: Boolean, optional (default=True)
        if to pop the Maxpooling layer of not
    
    Output:
    -------------
    model: Keras Sequential model 
        The full model, compiled, ready to be fit.
    '''
    
    #importing convolutional layers of vgg16 from keras
    model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
    if pop == True:
        model = Sequential(model.layers[:-1])
    #setting the convolutional layers to non-trainable 
    for layer in model.layers:
        layer.trainable = trainable
    return(model)
vgg_conv = get_VGG16()
vgg_conv.summary()
batch_size
2000//64
%%time
train_bottleneck_features = vgg_conv.predict_generator(generator=train_gen, steps=train_gen.n//batch_size +1 , verbose=1)
valid_bottleneck_features = vgg_conv.predict_generator(generator=valid_gen, steps=valid_gen.n//batch_size +1 , verbose=1)
train_bottleneck_features.shape
valid_bottleneck_features.shape
y_train = np_utils.to_categorical(train_gen.classes)
y_valid = np_utils.to_categorical(valid_gen.classes)

y_train = np.array([i[0] for i in y_train])
y_valid = np.array([i[0] for i in y_valid])
def top_model_vgg(n_classes,
                  X_shape,
                  dense_neurons=512,
                  do=0.5, 
                  loss_function = 'categorical_crossentropy',
                  output_activation='softmax',
                  optimizer='adam'):
    """ Top model multi:MLP 
    The top model corresponds to the VGG16's classification layers.
    The model is adapted for MULTICLASS classification tasks.
    
    Parameters:
    -------------
    n_classes: int 
        How many classes are you trying to classify ? 
        
    X_shape: tuple, optional (default=(7,7,512))
        The input shape for the first layer.
    
    dense_neurons: int, optional (default=512)
        The number of neurons in the hidden dense layers
        
    do: float, optional (default=0.5) 
        Dropout probability
    
    loss_function: str, optional (default='categorical_crossentropy')
        The loss function (keras object)
        
    output_activation: str, optional(default='softmax')
        The output activation layer
    
    optimizer: str or keras object, optional (default='adam')
        The optimizer of your choice.
        
    Output:
    -------------
    model: Keras Sequential model 
        The full model, compiled, ready to be fit.
    """
    
    ### top_model takes output from VGG conv and then adds 2 hidden layers
    top_model = Sequential()
    top_model.add(MaxPooling2D(input_shape=X_shape,name = 'top_maxpooling'))
    
    top_model.add(BatchNormalization())
    
    top_model.add(Flatten(name='top_flatten'))
    top_model.add(Dense(dense_neurons, activation='relu', name='top_relu_1'))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(do))
    
    top_model.add(Dense(dense_neurons, activation='relu', name='top_relu_2'))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(do))
    
    ### the last multilabel layers with the number of classes
    top_model.add(Dense(n_classes, activation=output_activation))
    
    top_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    
    return(top_model)
# choosing your optimizer
adam = keras.optimizers.Adam()
rmsprop = keras.optimizers.RMSprop(lr=0.01)

# building model
model = top_model_vgg(X_shape=(14,14,512),
                      dense_neurons=512,
                      do=0.75,
                      optimizer=adam,
                      n_classes=1,
                      loss_function='binary_crossentropy',
                      output_activation='sigmoid')
model.summary()
model.fit(train_bottleneck_features, y_train,
          epochs=4,
          batch_size=64,
          validation_data=(valid_bottleneck_features,y_valid))
model.fit(train_bottleneck_features, y_train,
          epochs=4,
          batch_size=64,
          validation_data=(valid_bottleneck_features,y_valid))

