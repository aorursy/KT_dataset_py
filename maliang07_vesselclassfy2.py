from __future__ import print_function, division

import numpy as np

import os

import glob

import cv2

import csv

import pandas as pd

from math import ceil

from PIL import Image, ImageFilter

#from scipy.misc import imresize, imsave

import gc

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss, confusion_matrix

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

#from PIL import Image, ImageChops, ImageOps



import matplotlib.pyplot as plt

from tensorflow.python.keras.optimizers import Adam, SGD

from tensorflow.python.keras.models import Model, Sequential

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.python.keras.utils import np_utils

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from tensorflow.python.keras.applications import DenseNet121, DenseNet201, DenseNet169

from tensorflow.python.keras.applications.nasnet import NASNetLarge, NASNetMobile

#from tensorflow.python.keras.models import Sequential, model_from_json

from tensorflow.python.keras import layers

from tensorflow.python.keras.applications.resnet50 import ResNet50

from tensorflow.python.keras.applications.inception_v3 import InceptionV3

from tensorflow.python.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Input

from tensorflow.python.keras.utils import to_categorical

#from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout

#from deepaugment.deepaugment import DeepAugment



val_split = .15  # if not using kfold cv

classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

NUM_CLASSES = len(classes)

FREEZE_LAYERS = 2

global img_width

global img_height

img_width=312

img_height=312





 

def crop(fname):

    #img = Image.open(fname)

    img = cv2.imread(fname) #直接读为灰度图像

    #img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # 合并每一个通道

    cropped = False

    #img = cv2.equalizeHist(img)

    #blurred = img.filter(ImageFilter.BLUR) 这里学长用了blur后的，可能可以更均匀的识别背景？ 我没有很明白

    #ba = np.array(blurred)

    if cropped:

        ba = np.array(img)

        h, w,_ = ba.shape

    #这里的1.2, 32, 5, 0.8都是后续可以调整的参数。 只是暂时觉得用这个来提取背景不错。

        left=0

        right=0

        uper=0

        low=0

        hor = ba[h//2, : ,1]

        vect = ba[:,w//2,1]

        for num1 in range(1,w):

            if hor[num1]>4:

                left = num1

                break

        for num2 in range(1,w):

            if hor[w-num2]>4:

                right = w-num2

                break

        for num3 in range(1,h):

            if vect[num3]>4:

                uper = num3

                break

        for num4 in range(1,h):

            if vect[h-num4]>4:

                low = h-num4

                break

        img = img[left:right,uper:low]

    #resized = cropped.resize([img_width, img_height])

    img = cv2.resize(img,(img_width,img_height),interpolation=cv2.INTER_CUBIC)

    return img



def load_train():

    n=3662

    X_train = np.zeros((n, img_width, img_height, 3), dtype=np.float32)

    y_train = np.zeros((n, 1), dtype=np.float32)

    #X_train = []

    #y_train = []

    i=0

    csvFile = open("../input/train.csv", "r")

    dict_reader = csv.DictReader(csvFile)

    print('Loading training images...')

    train_path='../input/train_images/'

    for row in dict_reader:

        flname = row['id_code'][0:12]

        path = os.path.join(train_path,flname+'.png')

        #print('path',path)

        #img_obj = crop(path)

        img = cv2.imread(path) #直接读为灰度图像

        img = cv2.resize(img,(img_width,img_height),interpolation=cv2.INTER_CUBIC)   

        #X_train[i] = img_array[sp[0]]

        X_train[i] = np.array(img) 

        y_train[i] = row['diagnosis']

        ## 图像预处理

        #dataset_normalized

        #clahe_equalized

        del img

        #adjust_gamma

        #读入图片 + 转成RGB + resize

        i = i+1

        #if i>=2500:

        #    break

        print('sp',i)

        #gc.collect()

        #print('y_train',y_train)

    gc.collect()

    #print('Training data load time: {} seconds'.format(round(time.time() - start_time, 2)))

    #print('y_train',y_train)

    #imgs_std = np.std(X_train)

    #imgs_mean = np.mean(X_train)

    #imgs_normalized = (X_train-imgs_mean)/imgs_std

    #for i in range(X_train.shape[0]):

    #    X_train[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255

    #return imgs_normalized

    return X_train, y_train





def build_model1():

    densenet = DenseNet169(

    #densenet = DenseNet121(

    weights='imagenet',

    #weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    #pretrained=True,

    input_shape=(img_width,img_height,3)

    )

    model = Sequential()

    model.add(densenet)

    model.add(Dropout(0.25))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.00001),

        #optimizer=SGD(),

        metrics=['accuracy']

    )

    

    return model



def build_model3():

    densenet = InceptionV3(

    weights='imagenet',

    #densenet = DenseNet121(

    #weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    #pretrained=True,

    input_shape=(img_width,img_height,3)

    )

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.00005),

        #optimizer=SGD(),

        metrics=['accuracy']

    )

    

    return model



def build_model4():

    densenet = NASNetMobile(

    weights='imagenet',

    #densenet = DenseNet121(

    #weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    #pretrained=True,

    input_shape=(img_width,img_height,3)

    )

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.0001),

        #optimizer=SGD(),

        metrics=['accuracy']

    )

    

    return model



def build_model5():

    densenet = ResNet50(

    weights='imagenet',

    #densenet = DenseNet121(

    #weights='../input/densenet-keras/DenseNet-BC-121-32-no-top.h5',

    include_top=False,

    #pretrained=True,

    input_shape=(img_width,img_height,3)

    )

    model = Sequential()

    model.add(densenet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='softmax'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=0.0005),

        #optimizer=SGD(),

        metrics=['accuracy']

    )

    

    return model



def build_model2():

    net = InceptionResNetV2(include_top=False,

                        weights='imagenet',

                        input_tensor=None,

                        input_shape=(img_width,img_height,3))

    x = net.output

    x = layers.GlobalAveragePooling2D()(x)

    #x = Dense(1024, activation='relu')(x)

    x = layers.Dropout(0.5)(x)

    output_layer = layers.Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

    net_final = Model(inputs=net.input, outputs=output_layer)

    #for layer in net_final.layers[:FREEZE_LAYERS]:

    #    layer.trainable = False

    #for layer in net_final.layers[FREEZE_LAYERS:]:

    #    layer.trainable = True

    net_final.compile(optimizer=Adam(lr=0.0005),

                  loss='binary_crossentropy', metrics=['accuracy'])

    return net_final

#print(net_final.summary())

net_final=build_model1()

#print('train_target',train_target)

#net_final=build_model1()

train_data, train_target = load_train()



my_config = {

    "child_epochs":10,

    "opt_samples":5

}

train_target = to_categorical(train_target)

#deepaug = DeepAugment(images=train_data, labels=train_target, config=my_config)

print(train_target.shape)



X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=val_split, random_state=8)

#print('X_train',X_train.shape())

#dataAugmentaion = ImageDataGenerator(rotation_range = 60, zoom_range = 0.1,featurewise_center=True,featurewise_std_normalization=True,

#                  horizontal_flip=True,vertical_flip=True,fill_mode = "reflect", shear_range = 0.1,zca_whitening=True,

#                  brightness_range=[0.8, 1.2])

dataAugmentaion = ImageDataGenerator(

        featurewise_std_normalization=True,

        zoom_range=0.15,  # set range for random zoom

        #shear_range = 0.1,

        height_shift_range = 0.05,

        width_shift_range = 0.1,

        #rotation_range = 10,

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True  # randomly flip images

    )





earlyStopping = EarlyStopping(

    monitor='val_loss',

    patience=15,

    verbose=1,

    mode='min'

)



reduceLROnPlateau = ReduceLROnPlateau(

    monitor='val_loss',

    factor=0.1,

    patience=5,

    min_lr=0.000001,

    verbose=1,

    mode='min'

)

# training the model

#dataAugmentaion.fit(X_train)



net_final.fit_generator(dataAugmentaion.flow(X_train, Y_train, batch_size = 16),

validation_data = (X_valid, Y_valid), steps_per_epoch=ceil(3662/16),callbacks=[earlyStopping, reduceLROnPlateau],

validation_steps = 0.2,epochs = 100, verbose=1)

import gc

del X_train

del X_valid

del Y_train

del Y_valid

del train_data

del train_target

gc.collect()

def load_test():

    n=1928

    X_test = np.zeros((n, img_width, img_height, 3), dtype=np.float32)

    #y_test = np.zeros((n, 1), dtype=np.float32)

    i=0

    csvFile = open("../input/test.csv", "r")

    dict_reader = csv.DictReader(csvFile)

    print('Loading testing images...')

    train_path='../input/test_images/'

    for row in dict_reader:

        flname = row['id_code'][0:12]

        path = os.path.join(train_path,flname+'.png')

        #print('path',path)

        #img_obj = crop(path)

        img = cv2.imread(path) #直接读为灰度图像

        img = cv2.resize(img,(img_width,img_height),interpolation=cv2.INTER_CUBIC)   

        #X_train[i] = img_array[sp[0]]

        X_test[i] = np.array(img) 

        #y_train[i] = row['diagnosis']

        del img

        #adjust_gamma

        #读入图片 + 转成RGB + resize

        i = i+1

        #if i>=2500:

        #    break

        print('sp',i)

        #gc.collect()

        #print('y_train',y_train)

    gc.collect()

    return X_test



test_data = load_test()
test_result = net_final.predict(test_data)

print(test_result)
y_test = np.zeros((1928, 1), dtype=np.int32)

for i in range(0,1928):

    label_predict = np.argmax(test_result[i,:])

    y_test[i] = label_predict



print(y_test)

submit = pd.read_csv('../input/sample_submission.csv')

submit['diagnosis'] = y_test

submit.to_csv('submission.csv', index=False)

submit.head()