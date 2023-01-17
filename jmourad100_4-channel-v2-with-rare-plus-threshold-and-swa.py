import os
import numpy as np
import pandas as pd
from PIL import Image
from zipfile import ZipFile
import matplotlib.pyplot as plt
import cv2
import math
from imgaug import augmenters as iaa
%matplotlib inline

import time
t_start = time.time()
os.listdir('../input')
PATH_BASE = '../input/'
PATH_TRAIN = PATH_BASE+'train/'

SHAPE = (299,299,4)
BATCH_SIZE = 30
EPOCHS = 10
raw_labels = pd.read_csv(PATH_BASE+'train.csv')
data_names = os.listdir(PATH_TRAIN)

#extract label names and labels array[{name: ,label:}]
labels = []
for name, label in zip(raw_labels['Id'],raw_labels['Target'].str.split(" ")):
    labels.append({
        'name':name,
        'label':label
    })
    
#Split data to train/dev set
from sklearn.model_selection import train_test_split
train_idx, test_idx = train_test_split(labels, test_size=0.2)
print('train: ' + str(len(train_idx)) + '\n'+ 'validation: ' + str(len(test_idx)))
y_cat_train_dic = {}
for icat in range(28):
    target = str(icat)
    y_cat_train_5 = np.array([int(target in ee['label']) for ee in train_idx])
    y_cat_train_dic[icat] = y_cat_train_5
up_sample = {}
for k in y_cat_train_dic:
    v = y_cat_train_dic[k].sum()
    up_sample[k] = np.round(v / len(train_idx), 5)
print(up_sample)
def plt_barh(x, y, title):
    fig, ax = plt.subplots(figsize=(15,7))
    width = 0.75
    ind = np.arange(len(up_sample))  # the x locations for the groups
    ax.barh(ind, y, width, color="blue")
    ax.set_yticks(ind+width/2)
    ax.set_yticklabels(x, minor=False)
    plt.title(title)
    for i, v in enumerate(y):
        ax.text(v, i , str(v), color='blue', fontweight='bold')
    plt.xlabel('x')
    plt.ylabel('y')
x = list(up_sample.keys())
y = list(up_sample.values())
plt_barh(x, y, 'data imbalance')
test = labels[10]
print(test); print(test['name']); print(test['label'])

fig, ax = plt.subplots(1,4,figsize=(12,12))
fig.tight_layout()

#Try different mix method
names = [n['name'] for n in np.random.choice(labels, 1)]
R = np.array(Image.open(PATH_TRAIN+names[0]+'_red.png'))
ax[0].imshow(R,cmap='Reds')
ax[0].set_title('R')
G = np.array(Image.open(PATH_TRAIN+names[0]+'_green.png'))
ax[1].imshow(G,cmap='Greens')
ax[1].set_title('G')
B = np.array(Image.open(PATH_TRAIN+names[0]+'_blue.png'))
ax[2].imshow(B,cmap='Blues')
ax[2].set_title('B')
Y = np.array(Image.open(PATH_TRAIN+names[0]+'_yellow.png'))
ax[3].imshow(Y,cmap='YlOrBr')
ax[3].set_title('Y')

BY = (B+Y)
BY[BY>255] = 255
RY = (R+Y)
RY[RY>255] = 255
GY = (G+Y)
GY[GY>255] = 255

IMG = np.stack((R, G, B) ,axis=-1)
IMG2 = np.stack((R, G, BY) ,axis=-1)
IMG3 = np.stack((RY, G, B) ,axis=-1)
IMG4 = np.stack((R, GY, B) ,axis=-1)
IMG = cv2.resize(IMG,(299,299))

fig2, ax2 = plt.subplots(2,2)
fig2.set_size_inches(12,12)
ax2[0,0].set_title('R,G,B')
ax2[0,0].imshow(IMG)
ax2[0,1].set_title('R,G,BY')
ax2[0,1].imshow(IMG2)
ax2[1,0].set_title('RY,G,B')
ax2[1,0].imshow(IMG3)
ax2[1,1].set_title('R,GY,B')
ax2[1,1].imshow(IMG4)
IMG.shape
#Define data_generator
class data_generator:
    
    def __init__(self):
        pass
    
    def batch_train(self, idx, batch_size, shape, augment=True):
        #extract eandom name and corresponding label
        while True:
            name_list = []
            label_list = []

            for n in np.random.choice(idx, batch_size):
                name_list.append(n['name'])
                int_label = list(map(int, n['label']))
                label_list.append(int_label)

            #batch_images = 提取images存成array, shape=(batch_size, shpae[0], shape[1], shpae[2]) = batch_images
            batch_images = np.zeros((batch_size, shape[0], shape[1], shape[2]))
            i = 0
            for name in name_list:
                image = self.load_img(name, shape)
                if augment:
                    image = self.augment(image)
                batch_images[i] = image
                i+=1

            #batch_labels = 提取labels轉換為multiple one-hot, shape=(batch_size, 28)
            batch_labels = np.zeros((batch_size, 28))
            j = 0
            for label in label_list:
                batch_labels[j][label] = 1
                j+=1

            yield batch_images, batch_labels
        
    def load_img(self, name, shape):
        R = np.array(Image.open(PATH_TRAIN+name+'_red.png'))
        G = np.array(Image.open(PATH_TRAIN+name+'_green.png'))
        B = np.array(Image.open(PATH_TRAIN+name+'_blue.png'))
        Y = np.array(Image.open(PATH_TRAIN+name+'_yellow.png'))
        image = np.stack((R, G, B, Y) ,axis=-1)
        image = cv2.resize(image, (shape[0], shape[1]))
        image = np.divide(image, 255)
        return image
    
    def augment(self, image):
        aug = iaa.OneOf([
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5)
        ])
        image = aug.augment_image(image)
        return image
from keras import applications
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras import regularizers
import tensorflow as tf
import keras.backend as K
K.clear_session()
THRESHOLD = 0.5

K_epsilon = K.epsilon()
def f1(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K_epsilon)
    r = tp / (tp + fn + K_epsilon)

    f1 = 2*p*r / (p+r+K_epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K_epsilon)
    r = tp / (tp + fn + K_epsilon)

    f1 = 2*p*r / (p+r+K_epsilon)
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)

def show_history(history):
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('f1')
    ax[1].plot(history.epoch, history.history["f1"], label="Train f1")
    ax[1].plot(history.epoch, history.history["val_f1"], label="Validation f1")
    ax[2].set_title('categorical_accuracy')
    ax[2].plot(history.epoch, history.history["categorical_accuracy"], label="Train categorical_accuracy")
    ax[2].plot(history.epoch, history.history["val_categorical_accuracy"], label="Validation categorical_accuracy")
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

def SortedDict(adict): 
    new_dict = {}
    ks = adict.keys() 
    ks = sorted(ks)
    for key in ks:
        new_dict[key] = adict[key]
    return new_dict
# load base model
INPUT_SHAPE = (299,299,3)
base_model = applications.InceptionResNetV2(include_top=False ,weights='imagenet', input_shape=INPUT_SHAPE)

for l in base_model.layers[::-1][:]: # enable training just .. Layers
    l.trainable = True

# Add top-model to base_model
def make_classifier_model(input_dim=(8,8,1536)):
    inp = Input(shape=input_dim)
    X = Conv2D(128, kernel_size=(3,3), activation='relu')(inp)
    X = MaxPooling2D(pool_size=(2, 2))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.25)(X)
    X = Conv2D(64, kernel_size=(1,1), activation='relu')(X)
    X = BatchNormalization()(X)
    X = Flatten()(X)  # this converts our 3D feature maps to 1D feature vectors
    X = Dense(512, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu')(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    X = Dense(28)(X)
    pred = Activation('sigmoid')(X)
    classifier_model = Model(inp, pred, name='classifier_model')
    return classifier_model

# Add 4-channdel input layers to base_model
def make_input_model(shape=SHAPE):
    inp = Input(shape=shape, name='input0')
    pred = Conv2D(3,kernel_size=1,strides=1,padding='same',activation='tanh',
                  kernel_regularizer=regularizers.l2(1e-4))(inp)
    input_model = Model(inp, pred, name='input_model')
    return input_model

# Create model piece
classifier_model = make_classifier_model()
input_model = make_input_model()

# Combine models
inp = Input(shape=SHAPE, name='inputs')
X = input_model(inp)
X = base_model(X)
pred = classifier_model(X)
model = Model(inp, pred, name='full_model')

model.summary()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stochastic Weight Averaging: https://arxiv.org/abs/1803.05407
Implementaton in Keras from user defined epochs assuming constant 
learning rate
Cyclic learning rate implementation in https://arxiv.org/abs/1803.05407 
not implemented
Created on July 4, 2018
@author: Krist Papadopoulos
"""

import keras

class SWA(keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i, layer in enumerate(self.model.layers):
                self.swa_weights[i] = (self.swa_weights[i] * \
                    (epoch - self.swa_epoch) + self.model.get_weights()[i]) \
                    /((epoch - self.swa_epoch)  + 1)  

        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

BestModelWeightsPath = 'BestModel.hdf5'
swa = SWA('Best_Weights.hdf5', int(EPOCHS * 0.9))
check_point = ModelCheckpoint(
    BestModelWeightsPath, monitor='val_f1', verbose=1,
    save_best_only=True, 
    mode='max',
)
reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.5, min_delta=0.0001, patience=10, verbose=1)
earlyStop = EarlyStopping(monitor='val_f1', mode='max', patience=30, verbose=1)
callbacks_list = [check_point, reduce_lr, earlyStop, swa]
from keras.metrics import categorical_accuracy

model.compile( loss=f1_loss, optimizer=Adam(1e-3), metrics=['categorical_accuracy', f1])
generator = data_generator()
train_generator = generator.batch_train(train_idx, BATCH_SIZE, SHAPE, augment=True)
validation_generator = generator.batch_train(test_idx, 620, SHAPE, augment=False)
history = model.fit_generator(
    train_generator,
    steps_per_epoch= len(train_idx) // BATCH_SIZE,
    validation_data= next(validation_generator),
    epochs=EPOCHS,
    verbose=1,
    callbacks=callbacks_list
)
show_history(history)
#Use this cell to read model & weight
model.load_weights('Best_Weights.hdf5')
n_list = np.arange(0.1,0.5,0.02)
for idx in test_idx[:3]:
    name0 = idx['name']
    print(idx)
    print(idx['name'])
    print(name0)
%%time
from tqdm import tqdm
TP_data = {}
FP_data = {}
FN_data = {}
F1_best = 0
F1_ther = 0
for threshold in tqdm(n_list):
    F1_sum = 0
    TP_datai = {}
    FP_datai = {}
    FN_datai = {}
    for i in range(28):
        TP_datai[i] = 0
        FP_datai[i] = 0
        FN_datai[i] = 0
    for idx in test_idx[:500]:
        name0 = idx['name']
        generator = data_generator()
        image = generator.load_img(name0, SHAPE)
        score_predict = model.predict(image[np.newaxis,:])
        score_predict = np.array(score_predict)[0]
        label_predict = np.arange(28)[score_predict>=threshold]
        true_label = idx['label']
        true_label = np.array(true_label).astype(int)
        label_predict = set(label_predict)
        true_label = set(true_label)
        
        TP = sum(1 for num in label_predict if num in true_label)
        FP = sum(1 for num in label_predict if not num in true_label)
        FN = sum(1 for num in true_label if not num in label_predict)
        TN = 28 - (TP+FP+FN)
        F1_sum += 2*TP/(2*TP+FN+FP)
        
        # count for acc for every label type
        for num in label_predict:
            if num in true_label:
                TP_datai[num] += 1
            if num not in true_label:
                FP_datai[num] += 1
        for num in true_label:
            if num not in label_predict:
                FN_datai[num] += 1
        
        
    if F1_sum>F1_best:
        F1_best = F1_sum
        F1_thre = threshold
        TP_data = TP_datai
        FP_data = FP_datai
        FN_data = FN_datai
        
    print('F1_score_sum: ', F1_sum, 'at threshold: ', threshold)
TP_data = SortedDict(TP_data)
FP_data = SortedDict(FP_data)
FN_data = SortedDict(FN_data)
print('F1_best ', F1_best, '  F1_thre ', F1_thre)
print('TP_data ', TP_data)
print('FP_data ', FP_data)
print('FN_data ', FN_data)
def dict_to_barh(dict_data, title):
    x = list(dict_data.keys())
    y = list(dict_data.values())
    return plt_barh(x, y, title)

dict_to_barh(TP_data, 'TP_data')
dict_to_barh(FP_data, 'FP_data')
dict_to_barh(FN_data, 'FN_data')
submit = pd.read_csv(PATH_BASE+'sample_submission.csv')
%%time
PATH_TRAIN = PATH_BASE+'test/'
generator = data_generator()
predicted = []

for name in tqdm(submit['Id']):
    image = generator.load_img(name, SHAPE)
    score_predict = model.predict(image[np.newaxis,:])[0]
    label_predict = np.arange(28)[score_predict>=F1_thre]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)
submit['Predicted'] = predicted
submit.to_csv('4 channel V2 with rare plus threshold.csv', index=False)
t_finish = time.time()
print(f"Kernel run time = {(t_finish-t_start)/3600} hours")
