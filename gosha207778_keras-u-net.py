import os

import sys

import random

import warnings



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from tqdm import tqdm

from itertools import chain

from skimage.io import imread, imshow, imread_collection, concatenate_images

from skimage.transform import resize

from skimage.morphology import label

import skimage.io as io

import skimage.transform as trans



from keras.models import *

from keras.layers import *

from keras.optimizers import *

from keras.callbacks import *

from keras import backend as K

from keras.utils.generic_utils import get_custom_objects



import tensorflow as tf



# Set some parameters

IMG_WIDTH = 128

IMG_HEIGHT = 128

IMG_CHANNELS = 3

TRAIN_PATH = '../input/stage1_train/'

TEST_PATH = '../input/stage1_test/'



warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42

random.seed = seed

np.random.seed = seed
# Get train and test IDs

train_ids = next(os.walk(TRAIN_PATH))[1]

test_ids = next(os.walk(TEST_PATH))[1]
# Get and resize train images and masks

X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):

    path = TRAIN_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_train[n] = img

    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    for mask_file in next(os.walk(path + '/masks/'))[2]:

        mask_ = imread(path + '/masks/' + mask_file)

        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 

                                      preserve_range=True), axis=-1)

        mask = np.maximum(mask, mask_)

    Y_train[n] = mask



# Get and resize test images

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

sizes_test = []

print('Getting and resizing test images ... ')

sys.stdout.flush()

for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    path = TEST_PATH + id_

    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    X_test[n] = img



print('Done!')
# Check if training data looks all right

ix = random.randint(0, len(train_ids))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()
# Define IoU metric

def mean_iou(y_true, y_pred):

    prec = []

    for t in np.arange(0.5, 1.0, 0.05):

        y_pred_ = tf.to_int32(y_pred > t)

        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)

        K.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([up_opt]):

            score = tf.identity(score)

        prec.append(score)

    return K.mean(K.stack(prec), axis=0)
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



        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))

        self.history.setdefault('iterations', []).append(self.trn_iterations)



        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        

        K.set_value(self.model.optimizer.lr, self.clr())
# Build U-Net model



inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

s = Lambda(lambda x: x / 255) (inputs)



conv1 = Conv2D(64, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(s)

conv1 = Conv2D(64, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv1)

pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)





conv2 = Conv2D(128, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(pool1)

conv2 = Conv2D(128, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv2)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)



conv3 = Conv2D(256, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(pool2)

conv3 = Conv2D(256, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv3)

pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)



conv4 = Conv2D(512, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(pool3)

conv4 = Conv2D(512, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv4)

drop4 = Dropout(0.5)(conv4)

pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)



conv5 = Conv2D(1024, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(pool4)

conv5 = Conv2D(1024, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv5)

drop5 = Dropout(0.5)(conv5)



up6 = Conv2D(512, 2, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))

merge6 = concatenate([drop4,up6], axis = 3)

conv6 = Conv2D(512, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(merge6)

conv6 = Conv2D(512, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv6)



up7 = Conv2D(256, 2, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))

merge7 = concatenate([conv3,up7], axis = 3)

conv7 = Conv2D(256, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(merge7)

conv7 = Conv2D(256, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv7)



up8 = Conv2D(128, 2, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))

merge8 = concatenate([conv2,up8], axis = 3)

conv8 = Conv2D(128, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(merge8)

conv8 = Conv2D(128, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv8)



up9 = Conv2D(64, 2, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))

merge9 = concatenate([conv1,up9], axis = 3)

conv9 = Conv2D(64, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(merge9)

conv9 = Conv2D(64, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv9)

conv9 = Conv2D(2, 3, activation = swish, padding = 'same', kernel_initializer = 'he_normal')(conv9)

conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)



model = Model(input = inputs, output = conv10)



model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [mean_iou, 'accuracy'])

model.summary()
# Fit model

earlystopper = EarlyStopping(patience=5, verbose=1)

checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

clr = CyclicLR(base_lr=1e-4, max_lr=0.006,

                        step_size=2000., mode='triangular2')



results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50, 

                    callbacks=[earlystopper, checkpointer, clr])
# Predict on train, val and test

model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)

preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)

preds_test = model.predict(X_test, verbose=1)



# Threshold predictions

preds_train_t = (preds_train > 0.5).astype(np.uint8)

preds_val_t = (preds_val > 0.5).astype(np.uint8)

preds_test_t = (preds_test > 0.5).astype(np.uint8)



# Create list of upsampled test masks

preds_test_upsampled = []

for i in range(len(preds_test)):

    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 

                                       (sizes_test[i][0], sizes_test[i][1]), 

                                       mode='constant', preserve_range=True))
# Perform a sanity check on some random training samples

ix = random.randint(0, len(preds_train_t))

imshow(X_train[ix])

plt.show()

imshow(np.squeeze(Y_train[ix]))

plt.show()

imshow(np.squeeze(preds_train_t[ix]))

plt.show()
# Perform a sanity check on some random validation samples

ix = random.randint(0, len(preds_val_t))

imshow(X_train[int(X_train.shape[0]*0.9):][ix])

plt.show()

imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))

plt.show()

imshow(np.squeeze(preds_val_t[ix]))

plt.show()
# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python

def rle_encoding(x):

    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []

    prev = -2

    for b in dots:

        if (b>prev+1): run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1

        prev = b

    return run_lengths



def prob_to_rles(x, cutoff=0.5):

    lab_img = label(x > cutoff)

    for i in range(1, lab_img.max() + 1):

        yield rle_encoding(lab_img == i)
new_test_ids = []

rles = []

for n, id_ in enumerate(test_ids):

    rle = list(prob_to_rles(preds_test_upsampled[n]))

    rles.extend(rle)

    new_test_ids.extend([id_] * len(rle))
# Create submission DataFrame

sub = pd.DataFrame()

sub['ImageId'] = new_test_ids

sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))

sub.to_csv('sub-dsbowl2018-1.csv', index=False)