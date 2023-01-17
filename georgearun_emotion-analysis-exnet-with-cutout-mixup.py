import numpy as np
import pandas as pd
import imageio
import pylab
import tensorflow as tf
import random
import os
import random
from PIL import Image
from tqdm import tqdm_notebook
from tensorflow import Tensor
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization,MaxPooling2D,\
                                    AveragePooling2D, Flatten, Dense, Add, ZeroPadding2D, concatenate, GlobalAveragePooling2D,Lambda,GlobalMaxPooling2D,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint,LearningRateScheduler
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import *

import matplotlib.pyplot as plt

import math 
from tensorflow.keras.optimizers import SGD
# Tensoflow implementation of eXnet from https://www.researchgate.net/publication/339324794_eXnet_An_Efficient_Approach_for_Emotion_Recognition_in_the_Wild
# !git clone https://github.com/bckenstler/CLR.git
# !git clone https://github.com/yu4u/cutout-random-erasing.git
with open("/kaggle/input/fer2013-6cat/Copy of fer2013_6cat.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size
num_classes = 6 #angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 64
epochs = 80

x_train, y_train, x_test, y_test = [], [], [], []
x_valid, y_valid = [],[]

for i in range(1,num_of_instances):
 #try:
  emotion, img, usage = lines[i].split(",")

  val = img.split(" ")
  pixels = np.array(val, 'float32')

  emotion = tf.keras.utils.to_categorical(emotion, num_classes)

  if 'Training' in usage:
        y_train.append(emotion)
        x_train.append(pixels)
  elif 'PublicTest' in usage:
        y_test.append(emotion)
        x_test.append(pixels)
  elif 'PrivateTest' in usage:
        y_valid.append(emotion)
        x_valid.append(pixels)

#data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_valid = np.array(x_valid, 'float32')
y_valid = np.array(y_valid, 'float32')


x_train /= 255 #normalize inputs between [0, 1]
x_test /= 255
x_valid /= 255


x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

x_valid = x_valid.reshape(x_valid.shape[0], 48, 48, 1)
x_valid = x_valid.astype('float32')
class MixupGenerator():
    def __init__(self, X_train, y_train, batch_size=32, alpha=0.2, shuffle=True, datagen=None):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) * self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
    

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser
# From Pyimagesearch Search Adrian Rosebrock

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.01, max_lr=0.0001, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())

# gen = ImageDataGenerator(            
#           featurewise_std_normalization=False,              
#                         rotation_range = 30, width_shift_range = 0.2,
#                         height_shift_range = 0.2, shear_range = 0.2,
#                          zoom_range = 0.15,horizontal_flip=True, 
#                     fill_mode="nearest")
# train_generator = gen.flow(x_train, y_train, batch_size=batch_size)


datagen = ImageDataGenerator(
    featurewise_std_normalization=False,              
                        rotation_range = 30, width_shift_range = 0.15,
                        height_shift_range = 0.15, shear_range = 0.15,
                         zoom_range = 0.15,horizontal_flip=True, 
                    fill_mode="nearest",
    preprocessing_function=get_random_eraser(v_l=0, v_h=10, p =0.3))

generator = MixupGenerator(x_train, y_train, alpha=0.6, batch_size=batch_size, datagen=datagen)()


def relu_bn(inputs: Tensor) -> Tensor:
    bn = BatchNormalization()(inputs)
    relu = ReLU()(bn)
    
    return relu


def create_convnet():
    input_shape = Input(shape=(48, 48, 1))

    initial = Conv2D(64, (3, 3),padding='same')(input_shape)
    x = relu_bn(initial)
    x = Conv2D(64, (3, 3),padding='same')(x)
    x = relu_bn(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    initial_block = BatchNormalization()(x)
    
    para_block_1 = Conv2D(128, (1, 1), padding='same')(initial_block)
    p_1 = relu_bn(para_block_1)
    p_1 = Conv2D(128, (3, 3), padding='same')(p_1)
    p_1 = relu_bn(p_1)
    
    para_block_2 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(initial_block)
    p_2 = Conv2D(128, (1, 1), padding='same')(para_block_2)
    p_2 = relu_bn(p_2)
    p_2 = Conv2D(128, (3, 3), padding='same')(p_2)
    p_2 = relu_bn(p_2)
    
    merged_para_1 = tf.keras.layers.concatenate([p_1, p_2], axis=1)
    merged_para_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(merged_para_1)
    
    para_block_3 = Conv2D(256, (1, 1), padding='same')(merged_para_1)
    p_21 = relu_bn(para_block_3)
    p_21 = Conv2D(256, (3, 3), padding='same')(p_21)
    p_21 = relu_bn(p_21)
    
    para_block_4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(merged_para_1)
    p_22 = Conv2D(256, (1, 1), padding='same')(para_block_4)
    p_22 = relu_bn(p_22)
    p_22 = Conv2D(256, (3, 3), padding='same')(p_22)
    p_22 = relu_bn(p_22)
    
    merged_para_2 = tf.keras.layers.concatenate([p_21, p_22], axis=1)
    merged_para_2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(merged_para_2)
    
    x = Conv2D(512, (1, 1), padding='same')(merged_para_2)
    x = relu_bn(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = Conv2D(512, (3, 3), padding='same')(x)
    x = relu_bn(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)   
    x = AveragePooling2D((2,2),strides=(1, 1))(x)
    
    final = GlobalAveragePooling2D()(x)

    final = Dense(512, activation='relu')(final)
    final = Dropout(0.4)(final)
    final = Dense(6, activation='softmax')(final)
    
    model = Model(input_shape, final)
    
    return model
model = create_convnet()
model.summary()
# tf.keras.utils.plot_model(
#     model, to_file='model.png', show_shapes=True, show_layer_names=True,
#     rankdir='TB', expand_nested=False,)
#pyimagesearch CyclicLR

clr = CyclicLR(
	mode="triangular",
	base_lr=0.01,
	max_lr=0.0001,
	step_size= 60 * (x_train.shape[0] // 64))
clr_step_size = int(4 * (len(x_train)/batch_size))
clr_triangular = CyclicLR(mode='triangular',base_lr=0.01, max_lr=0.00008,step_size=clr_step_size)

opt = tf.keras.optimizers.SGD(learning_rate=0.01, decay=4e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
model.fit(
          generator,
          epochs = 200,
          validation_data=(x_valid,y_valid),
          steps_per_epoch = len(x_train)//batch_size,
          shuffle = True, verbose=1,
          callbacks=[clr]
          )
predicted_test_labels = np.argmax(model.predict(x_test), axis=1)
test_labels = np.argmax(y_test, axis=1)
print ("Accuracy score = ", accuracy_score(test_labels, predicted_test_labels))
print ("precision score = ", precision_score(test_labels, predicted_test_labels , average='weighted'))
print ("recall score = ", recall_score(test_labels, predicted_test_labels, average='weighted'))
print ("f1 score = ", f1_score(test_labels, predicted_test_labels, average='weighted'))
train_score = model.evaluate(x_train, y_train, verbose=1)
print('Train loss:', train_score[0])
print('Train accuracy:', 100*train_score[1])

test_score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', test_score[0])
print('Test accuracy:', 100*test_score[1])  
# model.save("eXnet_64b_200e_SDG.h5")
len(x_train)
