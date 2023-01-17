import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
## Code of Wide Res Constructor source : https://github.com/titu1994/Wide-Residual-Networks 

from keras.models import Model
from keras.layers import Input, Add, Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

weight_decay = 0.0005

def initial_conv(input):
    x = Convolution2D(16, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(input)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    return x


def expand_conv(init, base, k, strides=(1, 1)):
    x = Convolution2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(init)

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = Convolution2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    skip = Convolution2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(init)

    m = Add()([x, skip])

    return m


def conv1_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv2_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def conv3_block(input, k=1, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
    x = Activation('relu')(x)
    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    if dropout > 0.0: x = Dropout(dropout)(x)

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)
    x = Convolution2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                      W_regularizer=l2(weight_decay),
                      use_bias=False)(x)

    m = Add()([init, x])
    return m

def create_wide_residual_network(input_dim, nb_classes=100, N=2, k=1, dropout=0.0, verbose=1):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    ip = Input(shape=input_dim)

    x = initial_conv(ip)
    nb_conv = 4

    x = expand_conv(x, 16, k)
    nb_conv += 2

    for i in range(N - 1):
        x = conv1_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 32, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv2_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = expand_conv(x, 64, k, strides=(2, 2))
    nb_conv += 2

    for i in range(N - 1):
        x = conv3_block(x, k, dropout)
        nb_conv += 2

    x = BatchNormalization(axis=channel_axis, momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
    x = Activation('relu')(x)

    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)

    x = Dense(nb_classes, W_regularizer=l2(weight_decay), activation='softmax')(x)

    model = Model(ip, x)

    if verbose: print("Wide Residual Network-%d-%d created." % (nb_conv, k))
    return model

if __name__ == "__main__":
    from keras.utils import plot_model
    from keras.layers import Input
    from keras.models import Model

   # init = (242, 162, 3)

    #wrn_28_10 = create_wide_residual_network(init, nb_classes=10, N=2, k=2, dropout=0.0)

    #wrn_28_10.summary()

    #plot_model(wrn_28_10, "WRN-16-2.png", show_shapes=True, show_layer_names=True)
    
   
# Creating the model of the training network

from keras import optimizers
from keras.models import load_model
init = (242, 162,3)
model =  create_wide_residual_network(init, nb_classes=10, N=2, k=2, dropout=0.4)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

## Loading initial weights
model.load_weights= ('../input/weights168/WRN-16-8 Weights.h5')

## Data augmentation block
# this blocks respresents all the data augmentation fucntions used

from keras.preprocessing import image
import tensorflow as tf
import scipy.signal
from tensorflow.python.ops import control_flow_ops
import random
import cv2

def recolor (img):
    img32 = img*255
    img_recolor =  img32/16
    img_recolor = img_recolor.astype(np.uint8)
    img_recolor = (img_recolor*16.0)/255
    return img_recolor

def sharper_image(img):
    img[:,:,0] = scipy.signal.convolve2d(img[:,:,0], np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))[1:243,1:163]
    img[:,:,1] = scipy.signal.convolve2d(img[:,:,1], np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))[1:243,1:163]
    img[:,:,2] = scipy.signal.convolve2d(img[:,:,2], np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))[1:243,1:163]
    return img

def average_mean_image (img):
    img_mean = (img - img.mean()) / img.std()
    return img_mean

def speckle_noise(img):
     row,col,ch = img.shape
     gauss = np.random.randn(row,col,ch)
     gauss = gauss.reshape(row,col,ch)        
     noisy = img + img * gauss
     return noisy
    
def rand_brightness(img):
    num=random.randint(1,80)-40
    X=np.ones(img.shape)
    X=X*num
    img=img*255+X
    img=img/255
    return img

def  preprocessing_function(img) :
    
    img = image.img_to_array(img) 

    R =  random.randint(1,9)
    if (R<3): 
        img = rand_brightness(img)
    
    R = random.randint(1,9)
    if (R<3) : 
        img = cv2.GaussianBlur(img,(5,5),0)
    elif(R<5): 
        img = cv2.GaussianBlur(img,(3,3),0)
    
    R = random.randint(1,9)
    if (R<3) : 
        img = speckle_noise(img)    
    R = random.randint(1,9)
    if (R<3) : 
        img = sharper_image(img)
 
    return img
    
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.3,
        width_shift_range = [-0.07,0,+0.07],
        height_shift_range = [-0.07,0,+0.07],
        horizontal_flip=False,
        vertical_flip=True,
        fill_mode = "reflect",
        preprocessing_function = preprocessing_function)

test_datagen = ImageDataGenerator(rescale=1./255,  
        zoom_range=0.3,
        width_shift_range = [-0.10,0,+0.10],
        height_shift_range = [-0.10,0,+0.10],
        horizontal_flip=False,
        vertical_flip=True,                                
        fill_mode = "reflect",
        preprocessing_function = preprocessing_function)

train_generator = train_datagen.flow_from_directory(
        '../input/data-resized/newdata/newData/train',
        target_size=(242, 162),
        batch_size=30,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        '../input/data-resized/newdata/newData/val',
        target_size=(242, 162),
        batch_size=30,
        class_mode='categorical')

history=  model.fit_generator(
        train_generator,
        steps_per_epoch=231,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=60)

from keras.models import model_from_json
model.save_weights('model_weights.h5')
0
# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

import random
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os

Testing_directory = '../input/testing-data/resized_test/resized_test'
#Testing_directory = '../input/data-resized/newdata/newData/train/0'
A = (os.listdir(Testing_directory)) 
sampled_list = random.sample(A,50)
predictions = [] 
for i in A:
    image_path = Testing_directory+'/'+i
    img = image.load_img(image_path, target_size=(242,162))
    x = image.img_to_array(img)
    x = (x/255.)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    K = [str(i)[:5], np.argmax(preds)+1]
    predictions.append(K)
predictions = pd.DataFrame(predictions)

print(predictions)
predictions.to_csv('out.csv', encoding='utf-8')
#x = preprocess_input(x)


#print(x)

print(history.history)