#所有包的导入

import json

import cv2

import random

import numpy as np

import tensorflow as tf

import time

import os

import matplotlib.pyplot as plt

%matplotlib inline



from tensorflow.keras import applications

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.utils import shuffle

from math import ceil

from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Sequential

from tensorflow.keras.applications import VGG16

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout

from tensorflow_core.python.keras.utils import np_utils

from tensorflow.keras.utils import plot_model
#训练参数设置  （config.py）    #需修改，改成你的参数及路径

EPOCHS = 10

BATCH_SIZE = 64

lEARN_RATE = 0.0001

CLASSES = 61



img_height = 224

img_width = 224   #输入样本的维度大小



train_num = 31718  #训练集样本数

val_num = 4507     #验证集样本数

test_num = 300      #测试集样本数



# TRAIN_DIR = "../input/mydataset/train/train/images/"

TRAIN_DIR = "../input/mydataclass/trainimg/trainimg/"



TRAIN_ANNOTATION_FILE = "../input/mydataset/train/train/AgriculturalDisease_train_annotations.json"



# VAL_DIR = "../input/dataset/val1/val/images/"

VAL_DIR = "../input/mydataclass/valimg/valimg/"



VAL_ANNOTATION_FILE = "../input/mydataclass/valimg/AgriculturalDisease_validation_annotations.json"

# build the VGG16_tl network

def vgg16_TL(input_shape=(img_width,img_height,3), classes=61):



    #实例化不含分类层的VGG16预训练模型，由include_top=False指定，

    #该模型是用ImageNet数据集训练的，由weights= " imagenet '指定

    # 实例化时会自动下载预训练权重



    model_base = applications.VGG16(weights='imagenet',

                                    include_top=False,

                                    input_shape=input_shape)

#     plot_model(model_base, to_file='vgg16.png', show_shapes=True)

    for layer in model_base.layers[:7]:   #[:15]锁住预训练模型的前15层

        layer.trainable = False  #训练期间不会更新层的权重

    model_base.summary()

    model = Sequential()

    #添加VGG16预训练模型

    model.add(model_base)

    #添加我们自己的分类层

#     model.add(GlobalAveragePooling2D())

    model.add(Flatten())



    model.add(Dense(1024, activation='relu'))  #4096

    model.add(Dropout(0.5))



    model.add(Dense(1024, activation='relu'))  #4096

    model.add(Dropout(0.5))



    model.add(Dense(classes, activation='softmax'))  #1000

    

#     model.summary()

    

    return model
#编译模型



model = vgg16_TL((224,224,3), classes=CLASSES)

sgd = SGD(lr=lEARN_RATE, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])



plot_model(model, to_file='model.png', show_shapes=True)
#图片进行图像预处理，增加图片归一化、适度旋转、随机缩放、上下翻转

train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    rotation_range=20,

    zoom_range=0.2,

    horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=(img_width, img_height),

    batch_size=BATCH_SIZE,

    class_mode='categorical') #多分类



validation_generator = val_datagen.flow_from_directory(

    VAL_DIR,

    target_size=(img_width, img_height),

    batch_size=BATCH_SIZE,

    class_mode='categorical') #多分类



# 训练模型

model_name= 'model_best.h5'



learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=1, verbose=1,factor=0.5, min_lr=0.000001)

checkpoint = ModelCheckpoint(model_name, monitor='val_accuracy', save_best_only=True)





tic = time.time()

history = model.fit_generator(train_generator,

                              steps_per_epoch=train_num // BATCH_SIZE,

                              epochs=EPOCHS,

                              validation_data=validation_generator,

                              validation_steps=val_num // BATCH_SIZE,

                              callbacks=[checkpoint, learning_rate_reduction])

toc = time.time()

print("")

print('used time:',toc - tic)   #可以输出从开始训练到结束所花费的时间 单位：秒

#保存模型

model.save_weights("Vgg16_Transfer_Learning_best.h5")
#保存训练性能图 （train.py）

title = "Crop Disease Identification Vgg16_TL Training Performance"    #需修改



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

fig.suptitle(title,fontsize=12)

fig.subplots_adjust(top=0.85,wspace=0.3)

epoch_list = list(range(1, EPOCHS +1))



ax1.plot(epoch_list,history.history['loss'], color='b', label="Training loss")

ax1.plot(epoch_list,history.history['val_loss'], color='r', label="validation loss")

ax1.set_xticks(np.arange(0, EPOCHS +1, 5))



ax1.set_ylabel('Loss Value')

ax1.set_xlabel('Epoch#')

ax1.set_title('Loss')

ax1.legend(loc="best")



ax2.plot(epoch_list,history.history['accuracy'], color='b', label="Training accuracy")

ax2.plot(epoch_list,history.history['val_accuracy'], color='r',label="Validation accuracy")

ax2.set_xticks(np.arange(0, EPOCHS +1, 5))



ax2.set_ylabel('Accuracy Value')

ax2.set_xlabel('Epoch#')

ax2.set_title('Accuracy')

ax2.legend(loc="best")



plt.savefig("Vgg16_TL_Training_Performance_best.png")   #需修改

plt.tight_layout()

plt.show()
loss = history.history['loss']

val_loss = history.history['val_loss']

accuracy = history.history['accuracy']

val_accuracy = history.history['val_accuracy']

print("loss = ", loss)

print("val_loss = ", val_loss)

print("accuracy = ", accuracy)

print("val_accuracy = ", val_accuracy)