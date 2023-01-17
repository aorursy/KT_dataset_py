import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
image_size = 256

def read_image(image_size=image_size):  # run this at the first time to read and save the images and labels as numpy array
    path = '../input/chest-xray-pneumonia/chest_xray/'
    image_set = []
    label_set = []
    for i in ['test','train','val']:
        for image_loc in os.listdir(path+i+'/NORMAL'):
            image = cv2.imread(path+i+'/NORMAL/'+image_loc,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image,(image_size,image_size))
            image = StandardScaler().fit_transform(image)
            image_set.append(image)
            label_set.append(0)
        for image_loc in os.listdir(path+i+'/PNEUMONIA'):
            image = cv2.imread(path+i+'/PNEUMONIA/'+image_loc,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image,(image_size,image_size))
            image = StandardScaler().fit_transform(image)
            image_set.append(image)
            label_set.append(1)
    image_set = np.array(image_set)
    image_set = np.expand_dims(image_set,axis=3)
    label_set = np.array(label_set)
    np.save('image_set_%s.npy'%image_size,image_set)
    np.save('label_set_%s.npy'%image_size,label_set)
    
# read_image(image_size)
path = '../input/building-an-inception-resnetv2-by-yourself/'
image = np.load(glob.glob(path+'image*.npy')[0])
label = np.load(glob.glob(path+'label*.npy')[0])
# np.save('image_set_%s.npy'%image_size,image)
# np.save('label_set_%s.npy'%image_size,label)
def split(image_set, label_set):
    x_train, x_test, y_train, y_test = train_test_split(image_set, label_set, train_size = 0.8, random_state = np.random)
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = split(image, label)
print(x_train.shape)
def datagen():    # for image augmentation but it slows the training process
    datagen = keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False, zca_epsilon=1e-06, rotation_range=20, width_shift_range=0.1,
        height_shift_range=0.1, brightness_range=None, shear_range=0.1, zoom_range=0.1,
        channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=True,
        vertical_flip=False, rescale=None, preprocessing_function=None, data_format='channels_last') 
    return datagen

train_datagen = datagen()
train_datagen.fit(x_train)
###################
filters = 32
kernel_size = (3,3)
stride = (1,1)
pool_size = (3,3)
###################
def conv2D(x,filters=filters,kernel=kernel_size,stride=stride,pad='same',activate=True,WN=False):
    if activate:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,padding=pad,activation='selu')(x)
    else:
        x = keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,strides=stride,padding=pad,activation=None)(x)
    if WN:
        tfa.addons.WeightNormalization(x)
    return x

def selu(x):
    x = tf.nn.selu(x)
    return x

def maxpool2D(x,pool_size=pool_size,stride=stride,pad='same'):
    x = keras.layers.MaxPool2D(pool_size=pool_size,strides=stride,padding=pad)(x)
    return x

def BN(x):
    x = keras.layers.BatchNormalization()(x)
    return x

def concat(x): # input as list
    x = tf.keras.layers.Concatenate()(x)
    return x

def res_add(raw_x,transformed_x,keep_scale):
    x = tf.keras.layers.Add()([raw_x*keep_scale,transformed_x*(1-keep_scale)])
    return x

def stem(x):
    x = keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2),input_shape=(image_size,image_size,1))(x)
    x = conv2D(x,64)
    x_1 = maxpool2D(x,(3,3),(2,2))                 # ---|
    x_2 = conv2D(x,96,(3,3),(2,2))                 # ---|
    x = concat([x_1,x_2])    # size/2, 160 channels
    x_1 = conv2D(x,64,(1,1))                       # ---|
    x_1 = conv2D(x_1,96,(3,3))                         #|
    x_2 = conv2D(x,64,(1,1))                           #|
    x_2 = conv2D(x_2,64,(1,7))                         #|
    x_2 = conv2D(x_2,64,(7,1))                         #|
    x_2 = conv2D(x_2,96,(3,3))                     # ---|
    x = concat([x_1,x_2])    # size/2, 192 channels                        
    x_1 = maxpool2D(x,(3,3),(2,2))                 # ---|
    x_2 = conv2D(x,192,(3,3),(2,2))                # ---|
    x = concat([x_1,x_2])    # 
    return x                 # size/2, 384 channels

def blockA(x):
    x_1 = x                  # highway
    x_2_1 = conv2D(x,32,(1,1))
    x_2_2 = conv2D(x,32,(1,1))
    x_2_2 = conv2D(x_2_2,32,(3,3))
    x_2_3 = conv2D(x,32,(1,1))
    x_2_3 = conv2D(x_2_3,48,(3,3))
    x_2_3 = conv2D(x_2_3,64,(3,3))
    x_2 = concat([x_2_1,x_2_2,x_2_3])
    x_2 = conv2D(x_2,384,(1,1),activate=False)
    x = res_add(x_1,x_2,0.2) # x_1 and x_2 must have the same number of channels to add up
    x = selu(x)
    return x                 # size fixed, channel fixed

def reduceA(x):
    x_1 = maxpool2D(x,(3,3),(2,2))  # size/2, channel fixed
    x_2 = conv2D(x,384,(3,3),(2,2))
    x_3 = conv2D(x,192,(1,1))
    x_3 = conv2D(x_3,192,(3,3))
    x_3 = conv2D(x_3,384,(3,3),(2,2))
    x = concat([x_1,x_2,x_3]) 
    return x                 # size/2, 1152 channel

def blockB(x):
    x_1 = x
    x_2_1 = conv2D(x,192,(1,1))
    x_2_2 = conv2D(x,128,(1,1))
    x_2_2 = conv2D(x_2_2,160,(1,7))
    x_2_2 = conv2D(x_2_2,192,(7,1))
    x_2 = concat([x_2_1,x_2_2])
    x_2 = conv2D(x_2,1152,(1,1),activate=False)
    x = res_add(x_1,x_2,0.2)
    x = selu(x)
    return x                 # size fixed, channel fixed

def reduceB(x):
    x_1 = maxpool2D(x,(3,3),(2,2))
    x_2 = conv2D(x,256,(1,1))
    x_2 = conv2D(x_2,384,(3,3),(2,2))
    x_3 = conv2D(x,256,(1,1))
    x_3 = conv2D(x_3,288,(3,3))
    x_3 = conv2D(x_3,320,(3,3),(2,2))
    x = concat([x_1,x_2,x_3])
    return x                 # size/2, 1856 channel

def blockC(x):
    x_1 = x
    x_2_1 = conv2D(x,192,(1,1))
    x_2_2 = conv2D(x,192,(1,1))
    x_2_2 = conv2D(x_2_2,224,(1,3))
    x_2_2 = conv2D(x_2_2,256,(3,1))
    x_2 = concat([x_2_1,x_2_2])
    x_2 = conv2D(x_2,1856,(1,1),activate=False)
    x = res_add(x_1,x_2,0.2)
    x = selu(x)
    return x                 # size fixed, channel fixed
    

def outputs(x):
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1, activation='sigmoid')(x)      # use sigmoid for binary classification
    return x

inputs = keras.Input(shape=(image_size,image_size,1))
x = stem(inputs)
x = blockA(x)
x = reduceA(x)
x = blockB(x)
x = blockB(x)
x = reduceB(x)
x = blockC(x)
outputs = outputs(x)

model = tf.keras.Model(inputs,outputs)
print(model.summary())
###################
total_epoch = 50
lr_init = 0.0001
batch_size = 8
###################

def scheduler(epoch):
    epoch += 1
    lr = lr_init
    threshold = 5
    depre = tf.math.exp(-0.25 * (epoch - threshold))
    if epoch <= threshold:
        return lr_init
    elif lr > lr_init/100:
        lr = lr_init * depre
        return lr
    else:
        return lr

scheduler = keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
# adding an earlystop so training process will stop in advance if the metrics (accuracy) doesn't improve for 5 epochs consecutively
earlystop = keras.callbacks.EarlyStopping(monitor="val_acc",mode="max",verbose=1,patience=5,restore_best_weights=True)
loss = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.Nadam(learning_rate=lr_init)
metrics = [tf.keras.metrics.BinaryAccuracy(name='acc')]
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
%%time

use_image_gen=False         # change to True if you want to use image data generator 

if use_image_gen:
    model.fit(train_datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=(len(x_train)/batch_size), epochs=total_epoch, callbacks=[callback],
    validation_data=(x_test, y_test), workers=0, use_multiprocessing=True, shuffle=True)
    
else:
    model.fit(x_train, y_train, batch_size=batch_size, epochs=total_epoch, callbacks=[scheduler,earlystop], validation_data=(x_test, y_test), shuffle=True)

model.save('model.h5')
pred = model.predict(x_test)
pred = np.round(pred,0)
print('confusion matrix:\n',metrics.confusion_matrix(y_test,pred))
print('precision:\n',metrics.precision_score(y_test,pred))
print('recall:\n',metrics.recall_score(y_test,pred))
print('f1_score:\n',metrics.f1_score(y_test,pred))
plt.title('model accuracy')
plt.plot(model.history.history['acc'],label='train accuracy')
plt.plot(model.history.history['val_acc'],label='test accuracy')
plt.legend()
plt.show()
plt.title('model loss')
plt.plot(model.history.history['loss'],label='train loss')
plt.plot(model.history.history['val_loss'],label='test loss')
plt.legend()
plt.show()