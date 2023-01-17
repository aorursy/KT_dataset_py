import os
import numpy as np
import pandas as pd
import pylab 
from PIL import Image
from IPython.display import SVG
import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)

%matplotlib inline
'''
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
'''

import math
import timeit
from six.moves import cPickle as pickle
import platform
#from subprocess import check_output
import glob

import tensorflow as tf
import keras
from keras.constraints import maxnorm
#from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils.np_utils import to_categorical   
from keras.utils.vis_utils import model_to_dot
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from tqdm import tqdm_notebook
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()
use_gpu
def unpickle(fname):
    with open(fname, "rb") as f:
        result = pickle.load(f, encoding='bytes')
    return result

def getData():
    labels_training = []
    dataImgSet_training = []
    labels_test = []
    dataImgSet_test = []

    # use "data_batch_*" for just the training set
    for fname in glob.glob("../input/cifar-10-batches-py/*data_batch*"):
        print("Getting data from:", fname)
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            #fname = data[b"filenames"][i]
            labels_training.append(data[b"labels"][i])

            # consecutive 1024 entries store color channels of 32x32 image 
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            
            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))  #Change the shape 3,32,32 to 32,32,3 
            dataImgSet_training.append(imgFormat)
            
    # use "test_batch_*" for just the test set
    for fname in glob.glob("../input/cifar-10-batches-py/*test_batch*"):
        print("Getting data from:", fname)
        data = unpickle(fname)

        for i in range(10000):
            img_flat = data[b"data"][i]
            #fname = data[b"filenames"][i]
            labels_test.append(data[b"labels"][i])

            # consecutive 1024 entries store color channels of 32x32 image 
            img_R = img_flat[0:1024].reshape((32, 32))
            img_G = img_flat[1024:2048].reshape((32, 32))
            img_B = img_flat[2048:3072].reshape((32, 32))
            
            imgFormat = np.array([img_R, img_G, img_B])
            imgFormat = np.transpose(imgFormat, (1, 2, 0))  #Change the shape 3,32,32 to 32,32,3 
            dataImgSet_test.append(imgFormat)
    
    
    dataImgSet_training = np.array(dataImgSet_training)
    labels_training = np.array(labels_training)
    dataImgSet_test = np.array(dataImgSet_test)
    labels_test = np.array(labels_test)
    
    return dataImgSet_training, labels_training, dataImgSet_test, labels_test
! ls ../input
! ls ../input/cifar-10-batches-py
X_train, y_train, X_test, y_test = getData()

labelNamesBytes = unpickle("../input/cifar-10-batches-py/batches.meta")
labelNames = []
for name in labelNamesBytes[b'label_names']:
    labelNames.append(name.decode('ascii'))

labelNames = np.array(labelNames)
fig = plt.figure(figsize=(6,6))
for i in range(0, 9):
    ax = fig.add_subplot(330 + 1 + i)
    plt.imshow(Image.fromarray(X_test[i]))
    ax.set_title(labelNames[y_test[i]])
    ax.axis('off')
    
plt.show()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]
datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(X_train)
num_classes = 10
input_shape = (32, 32, 3)
kernel = (3, 3)

# fix random seed for reproducibility 
seed = 101
np.random.seed(seed)
model = Sequential()
model.add(Conv2D(64, kernel_size=kernel, activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=kernel, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
SVG(model_to_dot(model, show_shapes=True, show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))
#training
batch_size = 50
epochs = 75
lrate = 0.1
epsilon=1e-08
decay=1e-4
#optimizer = keras.optimizers.rmsprop(lr=lrate,decay=1e-4)
optimizer = keras.optimizers.Adadelta(lr=lrate ) #, epsilon=epsilon, decay=decay)
#optimizer = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay)

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size, epochs=epochs, verbose=1,
                    validation_data=(X_test,y_test))
def plot_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epoch = range(epochs)

    fig = plt.figure(figsize=(20,6))
    ax1 = fig.add_subplot(121)
    plt.plot(epoch, acc, 'b', label='Training acc')
    plt.plot(epoch, val_acc, 'r', label='Validation acc')
    ax1.set_title('Training and validation accuracy')
    ax1.legend()

    ax2 = fig.add_subplot(122) 
    plt.plot(epoch, loss, 'b', label='Training loss')
    plt.plot(epoch, val_loss, 'r', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.legend()

    plt.show()

plot_results(history)
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print("Test Accuracy: %.2f%%" % (scores[1]*100))
#Saving the model
model.save('cifar10_1')
# How CNN Classifies an Image?
img_idx = 122
plt.imshow(X_test[img_idx],aspect='auto')
print('Actual label:', labelNames[np.argmax(y_test[img_idx])])
# Preper image to predict
test_image =np.expand_dims(X_test[img_idx], axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',labelNames[model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', model.predict_proba(test_image,batch_size=1))
# Utility Methods to understand CNN
# https://github.com/fchollet/keras/issues/431
def get_activations(model, model_inputs, print_shape_only=True, layer_name=None):
    import keras.backend as K
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations
# https://github.com/philipperemy/keras-visualize-activations/blob/master/read_activations.py
def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt

    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 1024:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        #plt.imshow(activations, interpolation='None', cmap='binary')
        fig, ax = plt.subplots(figsize=(18, 12))
        ax.imshow(activations, interpolation='None', cmap='binary')
        plt.show()
activations = get_activations(model, test_image)
display_activations(activations)
vgg_model = applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
display(vgg_model.summary())
bottleneck_path = r'../working/bottleneck_features_train_vgg19.npy'
# Set to false the layers except the last set of conv laer and their pooling
for layer in vgg_model.layers[:-5]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_model.layers:
    print(layer, layer.trainable)
# fix random seed for reproducibility
seed = 101
np.random.seed(seed)

# Create the model
clf_model = Sequential()
 
# Add the vgg convolutional base model
#clf_model.add(Flatten(input_shape=bottleneck_features_train.shape[1:]))
clf_model.add(vgg_model)

# Add new layers
clf_model.add(Flatten())
clf_model.add(Dense(1024, activation='relu'))
clf_model.add(Dropout(0.5))
clf_model.add(Dense(num_classes, activation='softmax'))

SVG(model_to_dot(clf_model, show_shapes=True, show_layer_names=True, rankdir='TB').create(prog='dot', format='svg'))
#training
batch_size = 50
epochs = 75
lrate = 0.1
epsilon=1e-08
decay=1e-4
#opt_rms = optimizers.rmsprop(lr=lrate,decay=1e-4)
optimizer = keras.optimizers.Adadelta(lr=lrate ) # decay=decay) #, epsilon=epsilon, 
#optimizer = keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=epsilon, decay=decay)

clf_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = clf_model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size, epochs=epochs, verbose=1,
                    validation_data=(X_test,y_test))
#history = clf_model.fit(bottleneck_features_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
plot_results(history)
scores = clf_model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print("Test Accuracy: %.2f%%" % (scores[1]*100))
img_idx = 177
plt.imshow(X_test[img_idx],aspect='auto')
print('Actual label:', labelNames[np.argmax(y_test[img_idx])])
# Preper image to predict
test_image =np.expand_dims(X_test[img_idx], axis=0)
print('Input image shape:',test_image.shape)
print('Predict Label:',labelNames[clf_model.predict_classes(test_image,batch_size=1)[0]])
print('\nPredict Probability:\n', clf_model.predict_proba(test_image,batch_size=1))