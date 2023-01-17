import numpy as np

import pandas as pd

import pydicom

import os

import collections

import sys

import glob

import random

import cv2

import tensorflow as tf

import multiprocessing



from math import ceil, floor

from copy import deepcopy

from tqdm import tqdm

from imgaug import augmenters as iaa



import keras

import keras.backend as K

from keras.callbacks import Callback, ModelCheckpoint

from keras.layers import Dense, Flatten, Dropout

from keras.models import Model, load_model

from keras.utils import Sequence

from keras.losses import binary_crossentropy

from keras.optimizers import Adam
# Install Modules from internet

!pip install efficientnet

!pip install iterative-stratification
# Import Custom Modules

import efficientnet.keras as efn 

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
# Seed

SEED = 11

np.random.seed(SEED)

tf.set_random_seed(SEED)



# Constants

TEST_SIZE = 0.06

HEIGHT = 300

WIDTH = 300

TRAIN_BATCH_SIZE = 24

VALID_BATCH_SIZE = 24



# Folders

DATA_DIR = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/'

TEST_IMAGES_DIR = DATA_DIR + 'stage_2_test/'

TRAIN_IMAGES_DIR = DATA_DIR + 'stage_2_train/'
def _get_first_of_dicom_field_as_int(x):

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def _get_windowing(data):

    dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]

    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]



def _window_image(img, window_center, window_width, slope, intercept):

    img = (img * slope + intercept)

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    return img 



def _normalize(img):

    if img.max() == img.min():

        return np.zeros(img.shape)

    return 2 * (img - img.min())/(img.max() - img.min()) - 1



def _read(path, desired_size=(224, 224)):

    dcm = pydicom.dcmread(path)

    window_params = _get_windowing(dcm) # (center, width, slope, intercept)



    try:

        # dcm.pixel_array might be corrupt (one case so far)

        img = _window_image(dcm.pixel_array, *window_params)

    except:

        img = np.zeros(desired_size)



    img = _normalize(img)



    if desired_size != (512, 512):

        # resize image

        img = cv2.resize(img, desired_size, interpolation = cv2.INTER_LINEAR)

    return img[:,:,np.newaxis]
# Image Augmentation

sometimes = lambda aug: iaa.Sometimes(0.25, aug)

augmentation = iaa.Sequential([  

                                iaa.Fliplr(0.25),

                                sometimes(iaa.Crop(px=(0, 25), keep_size = True, sample_independently = False))   

                            ], random_order = True)       

        

# Generators

class TrainDataGenerator(keras.utils.Sequence):



    def __init__(self, dataset, labels, batch_size=16, img_size=(512, 512), img_dir = TRAIN_IMAGES_DIR, augment = False, *args, **kwargs):

        self.dataset = dataset

        self.ids = dataset.index

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = img_size

        self.img_dir = img_dir

        self.augment = augment

        self.on_epoch_end()



    def __len__(self):

        return int(ceil(len(self.ids) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X, Y = self.__data_generation(indices)

        return X, Y



    def augmentor(self, image):

        augment_img = augmentation        

        image_aug = augment_img.augment_image(image)

        return image_aug



    def on_epoch_end(self):

        self.indices = np.arange(len(self.ids))

        np.random.shuffle(self.indices)



    def __data_generation(self, indices):

        X = np.empty((self.batch_size, *self.img_size, 3))

        Y = np.empty((self.batch_size, 6), dtype=np.float32)

        

        for i, index in enumerate(indices):

            ID = self.ids[index]

            image = _read(self.img_dir+ID+".dcm", self.img_size)

            if self.augment:

                X[i,] = self.augmentor(image)

            else:

                X[i,] = image            

            Y[i,] = self.labels.iloc[index].values        

        return X, Y

    

class TestDataGenerator(keras.utils.Sequence):

    def __init__(self, ids, labels, batch_size = 5, img_size = (512, 512), img_dir = TEST_IMAGES_DIR, *args, **kwargs):

        self.ids = ids

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = img_size

        self.img_dir = img_dir

        self.on_epoch_end()



    def __len__(self):

        return int(ceil(len(self.ids) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.ids[k] for k in indices]

        X = self.__data_generation(list_IDs_temp)

        return X



    def on_epoch_end(self):

        self.indices = np.arange(len(self.ids))



    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.img_size, 3))

        for i, ID in enumerate(list_IDs_temp):

            image = _read(self.img_dir+ID+".dcm", self.img_size)

            X[i,] = image            

        return X
def read_testset(filename = DATA_DIR + "stage_2_sample_submission.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df



def read_trainset(filename = DATA_DIR + "stage_2_train.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    duplicates_to_remove = [56346, 56347, 56348, 56349,

                            56350, 56351, 1171830, 1171831,

                            1171832, 1171833, 1171834, 1171835,

                            3705312, 3705313, 3705314, 3705315,

                            3705316, 3705317, 3842478, 3842479,

                            3842480, 3842481, 3842482, 3842483 ]

    df = df.drop(index = duplicates_to_remove)

    df = df.reset_index(drop = True)    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df



# Read Train and Test Datasets

test_df = read_testset()

train_df = read_trainset()
# Oversampling

epidural_df = train_df[train_df.Label['epidural'] == 1]

train_oversample_df = pd.concat([train_df, epidural_df])

train_df = train_oversample_df



# Summary

print('Train Shape: {}'.format(train_df.shape))

print('Test Shape: {}'.format(test_df.shape))
def predictions(test_df, model):    

    test_preds = model.predict_generator(TestDataGenerator(test_df.iloc[range(test_df.shape[0])].index, None, 5, (WIDTH, HEIGHT), TEST_IMAGES_DIR), verbose=1)

    return test_preds[:test_df.iloc[range(test_df.shape[0])].shape[0]]



def ModelCheckpointFull(model_name):

    return ModelCheckpoint(model_name, 

                            monitor = 'val_loss', 

                            verbose = 1, 

                            save_best_only = False, 

                            save_weights_only = True, 

                            mode = 'min', 

                            period = 1)



# Create Model

def create_model():

    K.clear_session()

    

    base_model =  efn.EfficientNetB3(weights = 'imagenet', 

                                     include_top = False, 

                                     pooling = 'avg', 

                                     input_shape = (HEIGHT, WIDTH, 3))

    x = base_model.output

    x = Dropout(0.125)(x)

    y_pred = Dense(6, activation = 'sigmoid')(x)



    return Model(inputs = base_model.input, outputs = y_pred)

# Radam Optimizer

from keras import backend as K

from keras.optimizers import Optimizer





# Ported from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py

class RectifiedAdam(Optimizer):

    """RectifiedAdam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments

        lr: float >= 0. Learning rate.

        final_lr: float >= 0. Final learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        gamma: float >= 0. Convergence speed of the bound function.

        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: Weight decay weight.

        amsbound: boolean. Whether to apply the AMSBound variant of this

            algorithm.

    # References

        - [On the Variance of the Adaptive Learning Rate and Beyond]

          (https://arxiv.org/abs/1908.03265)

        - [Adam - A Method for Stochastic Optimization]

          (https://arxiv.org/abs/1412.6980v8)

        - [On the Convergence of Adam and Beyond]

          (https://openreview.net/forum?id=ryQu7f-RZ)

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., weight_decay=0.0, **kwargs):

        super(RectifiedAdam, self).__init__(**kwargs)



        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.learning_rate = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')



        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay



        self.weight_decay = float(weight_decay)



    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.learning_rate

        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,

                                                      K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)



            beta2_t = self.beta_2 ** t

            N_sma_max = 2 / (1 - self.beta_2) - 1

            N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)



            # apply weight decay

            if self.weight_decay != 0.:

                p_wd = p - self.weight_decay * lr * p

            else:

                p_wd = None



            if p_wd is None:

                p_ = p

            else:

                p_ = p_wd



            def gt_path():

                step_size = lr * K.sqrt(

                    (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max /

                    (N_sma_max - 2)) / (1 - self.beta_1 ** t)



                denom = K.sqrt(v_t) + self.epsilon

                p_t = p_ - step_size * (m_t / denom)



                return p_t



            def lt_path():

                step_size = lr / (1 - self.beta_1 ** t)

                p_t = p_ - step_size * m_t



                return p_t



            p_t = K.switch(N_sma > 5, gt_path, lt_path)



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.learning_rate)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'weight_decay': self.weight_decay}

        base_config = super(RectifiedAdam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# loss function definition courtesy https://www.kaggle.com/akensert/resnet50-keras-baseline-model

from keras import backend as K



def logloss(y_true,y_pred):      

    eps = K.epsilon()

    

    class_weights = np.array([2., 1., 1., 1., 1., 1.])

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    #compute logloss function (vectorised)  

    out = -( y_true *K.log(y_pred)*class_weights

            + (1.0 - y_true) * K.log(1.0 - y_pred)*class_weights)

    return K.mean(out, axis=-1)



def _normalized_weighted_average(arr, weights=None):

    """

    A simple Keras implementation that mimics that of 

    numpy.average(), specifically for the this competition

    """

    

    if weights is not None:

        scl = K.sum(weights)

        weights = K.expand_dims(weights, axis=1)

        return K.sum(K.dot(arr, weights), axis=1) / scl

    return K.mean(arr, axis=1)



def weighted_loss(y_true, y_pred):

    """

    Will be used as the metric in model.compile()

    ---------------------------------------------

    

    Similar to the custom loss function 'weighted_log_loss()' above

    but with normalized weights, which should be very similar 

    to the official competition metric:

        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring

    and hence:

        sklearn.metrics.log_loss with sample weights

    """      

    

    eps = K.epsilon()

    

    class_weights = K.variable([2., 1., 1., 1., 1., 1.])

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    loss = -(y_true*K.log(y_pred)

            + (1.0 - y_true) * K.log(1.0 - y_pred))

    

    loss_samples = _normalized_weighted_average(loss,class_weights)

    

    return K.mean(loss_samples)
# Submission Placeholder

submission_predictions = []



# Multi Label Stratified Split stuff...

msss = MultilabelStratifiedShuffleSplit(n_splits = 20, test_size = TEST_SIZE, random_state = SEED)

X = train_df.index

Y = train_df.Label.values



# Get train and test index

msss_splits = next(msss.split(X, Y))

train_idx = msss_splits[0]

valid_idx = msss_splits[1]
data_generator_train = TrainDataGenerator(train_df.iloc[train_idx], 

                                                train_df.iloc[train_idx], 

                                                TRAIN_BATCH_SIZE, 

                                                (WIDTH, HEIGHT),

                                                augment = True)

    

data_generator_val = TrainDataGenerator(train_df.iloc[valid_idx], 

                                            train_df.iloc[valid_idx], 

                                            VALID_BATCH_SIZE, 

                                            (WIDTH, HEIGHT),

                                            augment = False)

from skimage import io



def imshow(image_RGB):

    io.imshow(image_RGB)

    io.show()



x1, y1 = data_generator_train[1]

x2, y2 = data_generator_val[1]

imshow(x1[0])

imshow(x2[0])
# Loop through Folds of Multi Label Stratified Split

#for epoch, msss_splits in zip(range(0, 9), msss.split(X, Y)): 

#    # Get train and test index

#    train_idx = msss_splits[0]

#    valid_idx = msss_splits[1]

for epoch in range(0, 3):

    print('=========== EPOCH {}'.format(epoch))



    # Shuffle Train data

    np.random.shuffle(train_idx)

    print(train_idx[:5])    

    print(valid_idx[:5])



    # Create Data Generators for Train and Valid

    data_generator_train = TrainDataGenerator(train_df.iloc[train_idx], 

                                                train_df.iloc[train_idx], 

                                                TRAIN_BATCH_SIZE, 

                                                (WIDTH, HEIGHT),

                                                augment = True)

    

    data_generator_val = TrainDataGenerator(train_df.iloc[valid_idx], 

                                            train_df.iloc[valid_idx], 

                                            VALID_BATCH_SIZE, 

                                            (WIDTH, HEIGHT),

                                            augment = False)



    # Create Model

    model = create_model()

    

    # Head Training Model

    if epoch < 1:

        model.load_weights('../input/rsna-models/model.h5')

        

    TRAIN_STEPS = int(len(data_generator_train) / 3)

    LR = 0.0001



    if epoch != 0:

        # Load Model Weights

        model.load_weights('model.h5')    



    model.compile(optimizer = RectifiedAdam(lr = LR), 

                  loss = 'binary_crossentropy',

                  metrics = [weighted_loss])

    

    # Train Model

    model.fit_generator(generator = data_generator_train,

                        validation_data = data_generator_val,

                        steps_per_epoch = TRAIN_STEPS,

                        epochs = 1,

                        callbacks = [ModelCheckpointFull('model.h5')],

                        verbose = 1)

    

    # Starting with epoch 4 we create predictions for the test set on each epoch

    if epoch > 1:

        preds = predictions(test_df, model)

        submission_predictions.append(preds)
test_df.iloc[:, :] = np.average(submission_predictions, axis = 0, weights = [2**i for i in range(len(submission_predictions))])

test_df = test_df.stack().reset_index()

test_df.insert(loc = 0, column = 'ID', value = test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

test_df.to_csv('submission.csv', index = False)