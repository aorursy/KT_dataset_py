import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2 # pic reading
import os
#print(os.listdir("../input"))

TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train/'
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) # read img into color mode
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = img-np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1,1,3))
    return img 
def prep_data(images):
    count = len(images)
    data = np.ndarray((count, 224, 224,3), dtype=np.float32)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image   
    return data
trainset = train_images[:5000]
validationset = train_images[-1000:]
train = prep_data(trainset)
validation = prep_data(validationset)
train_labels = []
for i in trainset:
    if i[i.find('train/')+6:i.find('train/')+9] =='dog':
        train_labels.append(1)
    else:
        train_labels.append(0)
val_labels = []
for i in validationset:
    if i[i.find('train/')+6:i.find('train/')+9] =='dog':
        val_labels.append(1)
    else:
        val_labels.append(0)    
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.layers import Flatten, Dense, Input
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
TF_WEIGHTS_PATH = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = '../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
def VGG_19(include_top=True, weights='imagenet',input_tensor=None):
    input_shape = (None, None, 3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    model = Model(img_input, x)
    if include_top:
        weights_path = TF_WEIGHTS_PATH
    else:
        weights_path = TF_WEIGHTS_PATH_NO_TOP
    model.load_weights(weights_path)
    return model
def save_bottlebeck_features():
    datagen = ImageDataGenerator()
    model = VGG_19(include_top=False, weights='imagenet')
    generator = datagen.flow(train,train_labels, batch_size=8,shuffle=False)
    bottleneck_features_train = model.predict_generator(generator, 625)
    np.save(open('bottleneck_features_train2.npy', 'wb'), bottleneck_features_train)
    print("bottleneck_train.npy is created..")
    
    generator = datagen.flow(validation,val_labels, batch_size=8,shuffle=False)
    bottleneck_features_validation = model.predict_generator(generator,125)
    np.save(open('bottleneck_features_validation2.npy', 'wb'), bottleneck_features_validation)
    print("bottleneck_validation.npy is created..")
save_bottlebeck_features()