from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import keras
import numpy as np
import pickle
from skimage import io , color
import colorsys
import numpy
import os
from ipywidgets import widgets
from keras.datasets.cifar import load_batch
from keras import backend as K
from ipywidgets import interact
np.random.seed(1337)
# remove file from kaggle working directory
converted_filenam =''
for files in os.listdir("/kaggle/working/") :
    if os.path.isfile("/kaggle/working/"+files):
        os.remove("/kaggle/working/"+files)
        
def load_data():
    """Loads CIFAR10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = '../input/cifar10-python/cifar-10-batches-py'
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000: i * 10000, :, :, :],
         y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def convertHSL(image):    
    for x in range(0, 32):
        for x1 in range(0, 32):
            image[x][x1][0] ,image[x][x1][1] ,image[x][x1][2]= colorsys.rgb_to_hls(image[x][x1][0] ,image[x][x1][1] ,image[x][x1][2])            
    return image

def srgb2rgb(c):    
    if (c > 0.03928):
       c= ((c + 0.055)/1.055)**2.4 
    else: c= c/12.92   
    return c

#linearized by inverting the gamma correction
def normalised_srgb(image):    
    for x in range(0, 32):
        for x1 in range(0, 32):
            image[x][x1][0] = srgb2rgb(image[x][x1][0])
            image[x][x1][1] = srgb2rgb(image[x][x1][1])
            image[x][x1][2] = srgb2rgb(image[x][x1][2])
    return image


def convert(colorspace='RGB' , subtract_pixelmean=False):
    num_classes = 10
    # Subtracting pixel mean improves accuracy True or False
    subtract_pixel_mean = subtract_pixelmean
        
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Input image dimensions.
    #input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    
    #conver to HSL
    if colorspace=="HSL" :
        for x in range(x_train.shape[0]):
            x_train[x]=convertHSL(x_train[x])
            
        for x1 in range(x_test.shape[0]):
            x_test[x1]=convertHSL(x_test[x1])

    #conver to HSV
    if colorspace=="HSV" :
        for x in range(x_train.shape[0]):
            x_train[x] = color.rgb2hsv(x_train[x])
            
        for x1 in range(x_test.shape[0]):
            x_test[x1] = color.rgb2hsv(x_test[x1])  
            
    #conver to XYZ
    if colorspace=="XYZ" :
        for x in range(x_train.shape[0]):
            x_train[x] = color.rgb2xyz(x_train[x])          
        for x1 in range(x_test.shape[0]):
            x_test[x1] = color.rgb2xyz(x_test[x1])  
                
        
    #conver to LUV
    if colorspace=="LUV" :
        for x in range(x_train.shape[0]):
            x_train[x] = color.rgb2luv(x_train[x]) 
            
        for x1 in range(x_test.shape[0]):
            x_test[x1] = color.rgb2luv(x_test[x1])  
    #conver to LAB
    if colorspace=="LAB" :
        for x in range(x_train.shape[0]):
            x_train[x] = color.rgb2lab(x_train[x]) 
            
        for x1 in range(x_test.shape[0]):
            x_test[x1] = color.rgb2lab(x_test[x1])  
            
    #conver to YUV
    if colorspace=="YUV" :
        for x in range(x_train.shape[0]):
            x_train[x] = color.rgb2yuv(x_train[x]) 
            
        for x1 in range(x_test.shape[0]):
            x_test[x1] = color.rgb2yuv(x_test[x1])
            
    
    filenam=colorspace
    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        filenam=filenam+"subtract_pixel_mean"
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)
    
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    filenam = filenam + "cifar10_normalized.pkl"
    print(filenam)
    with open(filenam, 'wb') as f:
        pickle.dump(((x_train, y_train), (x_test, y_test)), f)
    print("finish")
    return filenam

@interact
def show_articles_more_than(colorspace=['RGB', 'HSL', 'HSV', 'LAB','LUV','YUV'], subtract_pixel_mean=False):
    global converted_filenam
    converted_filenam = convert(colorspace, subtract_pixel_mean)
    return converted_filenam 



import pickle
import matplotlib.pyplot as plt  
with open('/kaggle/working/'+converted_filenam, 'rb') as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f)    
plt.imshow(x_train[7])
plt.show()
#io.imsave(converted_filenam+'B1_7.png',x_train[7])
from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint , TensorBoard
#callbacks=callbacks ,steps_per_epoch=391
import numpy as np
import os
import pickle
batch_size = 128 #@param {type:"slider", min:0, max:1024, step:32}

epochs = 200
data_augmentation = True #@param {type:"boolean"}
num_classes = 10
savemodel =""
n = 3

version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
# Load the CIFAR10 data.
path="/kaggle/working/"
with open(path+converted_filenam, 'rb') as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f) 
# Input image dimensions.
input_shape = x_train.shape[1:]
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model
model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)
# Prepare callbacks for model saving and for learning rate adjustment and log tensorboard
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
savemodel = converted_filenam+model_name
checkpoint = ModelCheckpoint(filepath=savemodel,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True , mode='max')



lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

mycallbacks = [checkpoint, lr_reducer, lr_scheduler]

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    history =model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=mycallbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    history =model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, callbacks=mycallbacks, workers=4)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

from pickle import dump 
with open('History'+savemodel, 'wb') as handle: # saving the history of the model
    dump(history.history, handle)
print("finich")
# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()
print(filters)
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    print("Filters of hidden Layers")
    print(f)
    # plot each channel separately
    for j in range(3):
        # specify subplot and turn of axis
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
# show the figure
pyplot.show()
#Printing filters
import numpy as np
import cv2
from PIL import Image
filters, biases = model.layers[1].get_weights()
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
    # get the filter
    f = filters[:, :, :, i]
    for j in range(3):
        # specify subplot and turn of axis
        ax = pyplot.subplot(n_filters, 3, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
# show the figure
pyplot.show()
from numpy import array
print(filters.shape)
#Taking only first channel
#Each channel has 16 filters and each filter is 3X3 with 3 channels
for t in range(16):
    z=filters[2:,:,:,-t]
    for j in range(2):
        #Reshaping into 3X3 Filters by taking only 1 channel of input
        q=numpy.reshape(z, (3,3))
        a=np.fft.fftshift(np.abs(np.fft.fft2(q))**2)
        pyplot.imshow(np.log(a))
    pyplot.show()
