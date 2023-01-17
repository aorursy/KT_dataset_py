import numpy as np 
import pandas as pd
import glob
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import Model,layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
paths=glob.glob('../input/intel-image-classification/seg_train/seg_train/*')
l=len('../input/intel-image-classification/seg_train/seg_train/')
labels=[]
for path in paths:
    labels.append(path[l:])
print(labels)
def prepare_dataset(path,label):
    x_train=[]
    y_train=[]
    all_images_path=glob.glob(path+'/*.jpg')
    for img_path in all_images_path :
            img=load_img(img_path, target_size=(150,150))
            img=img_to_array(img)
            img=img/255.0
            x_train.append(img)
            y_train.append(label)
    return np.array(x_train),np.array(y_train)
trainX_building, trainY_building  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/buildings/",0)
trainX_forest,trainY_forest  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/forest/",1)
trainX_glacier,trainY_glacier  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/glacier/",2)
trainX_mount,trainY_mount  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/mountain/",3)
trainX_sea,trainY_sea  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/sea/",4)
trainX_street,trainY_street  = prepare_dataset("../input/intel-image-classification/seg_train/seg_train/street/",5)

print('train building shape ', trainX_building.shape, trainY_building.shape) 
print('train forest', trainX_forest.shape ,trainY_forest.shape)
print('train glacier', trainX_glacier.shape,trainY_glacier.shape)
print('train mountain', trainX_mount.shape, trainY_mount.shape)
print('train sea',     trainX_sea.shape, trainY_sea.shape)
print('train street', trainX_street.shape ,trainY_street.shape)
x_train=np.concatenate((trainX_building,trainX_forest,trainX_glacier,trainX_mount,trainX_sea,trainX_street),axis=0)
y_train=np.concatenate((trainY_building,trainY_forest,trainY_glacier,trainY_mount,trainY_sea,trainY_street),axis=0)
print(x_train.shape)
print(y_train.shape)
testX_building, testY_building  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/buildings/",0)
testX_forest,testY_forest  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/forest/",1)
testX_glacier,testY_glacier  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/glacier/",2)
testX_mount,testY_mount  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/mountain/",3)
testX_sea,testY_sea  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/sea/",4)
testX_street,testY_street  = prepare_dataset("../input/intel-image-classification/seg_test/seg_test/street/",5)

x_test=np.concatenate((testX_building,testX_forest,testX_glacier,testX_mount,testX_sea,testX_street),axis=0)
y_test=np.concatenate((testY_building,testY_forest,testY_glacier,testY_mount,testY_sea,testY_street),axis=0)
local_weights_file = '/kaggle/input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
     layer.trainable = False
        
pre_trained_model.summary()
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)           

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])
history=model.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))
from tensorflow.keras.applications import ResNet50

pretrained_model=ResNet50( input_shape=(150,150,3),
                                  include_top=False,
                                  weights='imagenet'
                                   )

for layer in pretrained_model.layers:
     layer.trainable = False

pretrained_model.summary()
last_layer = pretrained_model.get_layer('conv5_block3_out')
print('last layer of vgg : output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)

model_resnet = Model(pretrained_model.input, x) 
model_resnet.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])
model_resnet.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))
from tensorflow.keras.applications import VGG16
file='/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
pretrained_model=VGG16(input_shape = (150, 150, 3), 
                        include_top = False, 
                        weights =None)

pretrained_model.load_weights(file)

for layer in pretrained_model.layers:
     layer.trainable = False
last_layer = pretrained_model.get_layer('block5_pool')
print('last layer of vgg : output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)                  
x = layers.Dense(6, activation='softmax')(x)           

model_vgg = Model(pretrained_model.input, x) 
model_vgg.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['acc'])

#model_vgg.fit(x_train,y_train,epochs=1,validation_data=(x_test,y_test))