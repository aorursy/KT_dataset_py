# Quoting https://raw.githubusercontent.com/bckenstler/CLR/master/clr_callback.py
from keras.callbacks import *

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

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
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
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
            

  
# Quoting https://raw.githubusercontent.com/yu4u/mixup-generator/master/random_eraser.py
import numpy as np

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
# Quoting https://raw.githubusercontent.com/yu4u/mixup-generator/master/mixup_generator.py
import numpy as np


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
# Preparation
import os
import shutil
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

# quoted above  from mixup_generator import MixupGenerator
# quoted above  from random_eraser import get_random_eraser
# quoted above  from cyclic_lr import CyclicLR

import keras
import keras.backend as K
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Activation, Dropout, BatchNormalization, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

TRY = 'E4' # Attempt Id
# Load dataset
datadir = "../input/"

X_train_org = np.load(datadir + 'X_train.npy')
X_test = np.load(datadir + 'X_test.npy')
y_labels_train = pd.read_csv(datadir + 'y_train.csv', sep=',')['scene_label'].tolist()
# Make label lists
labels = sorted(list(set(y_labels_train)))
label2int = {l:i for i, l in enumerate(labels)}
int2label = {i:l for i, l in enumerate(labels)}
# Map y_train to int labels
y_train_org = keras.utils.to_categorical([label2int[l] for l in y_labels_train])

# Train/Validation split to X_train/y_train
splitlist = pd.read_csv(datadir + 'crossvalidation_train.csv', sep=',')['set'].tolist()
X_train = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'train'])
X_valid = np.array([x for i, x in enumerate(X_train_org) if splitlist[i] == 'test'])
y_train = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'train'])
y_valid = np.array([y for i, y in enumerate(y_train_org) if splitlist[i] == 'test'])

# Subtract sample-wise mean = pseudo background subtraction
X_train = np.array([X - np.mean(X, axis=1, keepdims=True) for X in X_train])
X_valid = np.array([X - np.mean(X, axis=1, keepdims=True) for X in X_valid])
X_test = np.array([X - np.mean(X, axis=1, keepdims=True) for X in X_test])

# Normalize dataset
value_max = np.max(np.vstack([X_train, X_valid, X_test]))
X_train = X_train / value_max
X_valid = X_valid / value_max
X_test = X_test / value_max

# [:, 40, 501] -> [:, 40, 501, 1]
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Confirmation
def plot_dataset(X, n=3):
    for i in range(n):
        x = X[i]
        plt.pcolormesh(x[..., -1])
        plt.show()
for X in [X_train, X_valid, X_test]:
    plot_dataset(X, n=1)
batch_size = 32 
num_classes = len(labels)
epochs = 10 # CHANGE TO 250, 20 is just for Kaggle kernel

datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.4,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    preprocessing_function=get_random_eraser(v_l=np.min(X_train), v_h=np.max(X_train)) # RANDOM ERASER
)
mixupgen = MixupGenerator(X_train, y_train, alpha=1.0, batch_size=batch_size, datagen=datagen)
test_datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
)

datagen.fit(np.r_[X_train, X_valid, X_test])
test_datagen.mean, test_datagen.std = datagen.mean, datagen.std

# Visualize some of them.
tmp = next(mixupgen())
for X in [tmp[0][:5]]:
    plot_dataset(X)
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, GlobalMaxPooling2D, AveragePooling2D, Activation, Dropout, BatchNormalization, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard

#  Based on [TensorFlow Speech Recognition Challenge 2nd place solution by Thomas O'Malley
# https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/discussion/47715
def model_cnn7(input_shape):
    model = Sequential()
    #  > I thought of this as a "denoising" and basic feature extraction step
    model.add(Conv2D(64, (7, 3),  input_shape=input_shape, strides=(1,2), activation='relu', padding='same'))
    # > Getting back down to the standard 40 frequency features
    model.add(MaxPooling2D((4, 1), strides=(2, 1)))
    model.add(BatchNormalization())
    #  > Look for local patterns across frequency bands
    model.add(Conv2D(128, (7, 1),  strides=(1,1), activation='relu', padding='same'))
    # Allow for speaker variation, similar to what worked here:
    # https://link.springer.com/content/pdf/10.1186%2Fs13636-015-0068-3.pdf
    model.add(MaxPooling2D((4, 2), strides=(2, 2)))
    model.add(BatchNormalization())
    # This allows it to treat each remaining freq band very differently, and compress the frequency dimension entirely.
    # I think of this as detecting phoneme-level features
    model.add(Conv2D(128, (5, 1),  input_shape=input_shape, strides=(1,1), activation='relu', padding='valid'))
    model.add(BatchNormalization())
    # I think of this as looking for connected components of a short keyword at different points in time
    model.add(Conv2D(128, (1, 5),  input_shape=input_shape, strides=(1,1), activation='relu', padding='same'))
    # Collect all the components
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(0.25))
    # Because why not, and seemed to work well
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = model_cnn7(X_train.shape[1:])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])
# Prepare callback
callbacks = [
    # Cyclic Learning Rate
    CyclicLR(base_lr=0.00007, max_lr=0.0007, step_size=X_train.shape[0] // batch_size, mode='triangular'),
    # Save the best model
    ModelCheckpoint('weights%s4valid.model' % TRY,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only=True),
     keras.callbacks.TensorBoard(log_dir='./%slog' % TRY, histogram_freq=0, write_graph=True, write_images=True)
]

# Fit the model on the batches generated by mixupgen().
model.fit_generator(mixupgen(),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=epochs,
                    validation_data=test_datagen.flow(X_valid, y_valid), callbacks=callbacks)
# Fine tune
model.load_weights('weights%s4valid.model' % TRY)
K.set_value(model.optimizer.lr, 0.000001)
model.fit_generator(mixupgen(),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=10,
                    validation_data=test_datagen.flow(X_valid, y_valid), callbacks=None)
model.save_weights('weights%s4valid.model' % TRY)
# CONFIRMATION ON INITIAL TRAINING
model.load_weights('weights%s4valid.model' % TRY)
y_valid_preds = model.predict_generator(test_datagen.flow(X_valid, y_valid, shuffle=False))
y_valid_pred_cls = [np.argmax(pred) for pred in y_valid_preds]
y_valid_refs = [np.argmax(y) for y in y_valid]

np.save('preds%s4valid.npy' % TRY, y_valid_preds)
valid_results = [result == ref for result, ref in zip(y_valid_pred_cls, y_valid_refs)]

# accuracy
print(np.sum(valid_results)/len(valid_results))

# double check the answers
for result, ref in zip(y_valid_pred_cls[:10], y_valid_refs[:10]):
    print(result, '\t', ref)
