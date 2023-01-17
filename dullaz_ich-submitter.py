import os

import json



import cv2

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import pydicom

import cupy as cp



from keras import layers

from keras.applications import DenseNet121, ResNet50V2, InceptionV3

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator

from keras.initializers import Constant

from keras.utils import Sequence

from keras.models import Sequential

from keras.optimizers import Adam

from keras.models import Model, load_model

from keras.layers import GlobalAveragePooling2D, Dense, Activation, concatenate, Dropout

from keras.initializers import glorot_normal, he_normal

from keras.regularizers import l2



import keras.metrics as M

import tensorflow_addons as tfa

import pickle



from keras import backend as K



import tensorflow as tf

from tensorflow.python.ops import array_ops



from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold



import warnings

warnings.filterwarnings(action='once')
test_csv = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"

test_dir = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test"

BASE_PATH = "../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/"

TEST_DIR = "stage_2_test/"

test_df = pd.read_csv(test_csv)

test_df.head()
test_df.shape
testdf = test_df.ID.str.rsplit("_", n=1, expand=True)

testdf = testdf.rename({0: "id", 1: "subtype"}, axis=1)

testdf.loc[:, "label"] = 0

testdf.head()

testdf.shape
testdf = pd.pivot_table(testdf, index="id", columns="subtype", values="label")

testdf.head(1)

testdf.shape
def sigmoid_window(dcm, window_center, window_width, U=1.0, eps=(1.0 / 255.0)):

    img = dcm.pixel_array

    img = cp.array(np.array(img))

    _, _, intercept, slope = get_windowing(dcm)

    img = img * slope + intercept

    ue = cp.log((U / eps) - 1.0)

    W = (2 / window_width) * ue

    b = ((-2 * window_center) / window_width) * ue

    z = W * img + b

    img = U / (1 + cp.power(np.e, -1.0 * z))

    img = (img - cp.min(img)) / (cp.max(img) - cp.min(img))

    return cp.asnumpy(img)

def get_first_of_dicom_field_as_int(x):

    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)

    if type(x) == pydicom.multival.MultiValue:

        return int(x[0])

    else:

        return int(x)



def get_windowing(data):

    dicom_fields = [data[('0028','1050')].value, #window center

                    data[('0028','1051')].value, #window width

                    data[('0028','1052')].value, #intercept

                    data[('0028','1053')].value] #slope

    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def preprocess(file,type="WINDOW",DIR=TEST_DIR):

    dcm = pydicom.dcmread(BASE_PATH+DIR+file+".dcm")

    if type == "WINDOW":

        window_center , window_width, intercept, slope = get_windowing(dcm)

        w = window_image(dcm, window_center, window_width)

        win_img = np.repeat(w[:, :, np.newaxis], 3, axis=2)

        #return win_img

    elif type == "SIGMOID":

        window_center , window_width, intercept, slope = get_windowing(dcm)

        test_img = dcm.pixel_array

        w = sigmoid_window(dcm, window_center, window_width)

        win_img = np.repeat(w[:, :, np.newaxis], 3, axis=2)

        #return win_img

    elif type == "BSB":

        win_img = bsb_window(dcm)

        #return win_img

    elif type == "SIGMOID_BSB":

        win_img = sigmoid_bsb_window(dcm)

    elif type == "GRADIENT":

        win_img = rainbow_window(dcm)

        #return win_img

    else:

        win_img = dcm.pixel_array

    resized = cv2.resize(win_img,(224,224))

    return resized



class DataLoader(Sequence):

    def __init__(self, dataframe,

                 batch_size,

                 shuffle,

                 input_shape,

                 num_classes=6,

                 steps=None,

                 prep="SIGMOID"):

        

        self.data_ids = dataframe.index.values

        self.dataframe = dataframe

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.input_shape = input_shape

        self.num_classes = num_classes

        self.current_epoch=0

        self.prep = prep

        self.steps=steps

        if self.steps is not None:

            self.steps = np.round(self.steps/3) * 3

            self.undersample()

        

    def undersample(self):

        part = np.int(self.steps/3 * self.batch_size)

        zero_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 0].index.values, size=5000, replace=False)

        hot_ids = np.random.choice(self.dataframe.loc[self.dataframe["any"] == 1].index.values, size=5000, replace=True)

        self.data_ids = list(set(zero_ids).union(hot_ids))

        np.random.shuffle(self.data_ids)

        

    # defines the number of steps per epoch

    def __len__(self):

        if self.steps is None:

            return np.int(np.ceil(len(self.data_ids) / np.float(self.batch_size)))

        else:

            return 3*np.int(self.steps/3) 

    

    # at the end of an epoch: 

    def on_epoch_end(self):

        # if steps is None and shuffle is true:

        if self.steps is None:

            self.data_ids = self.dataframe.index.values

            if self.shuffle:

                np.random.shuffle(self.data_ids)

        else:

            self.undersample()

        self.current_epoch += 1

    

    # should return a batch of images

    def __getitem__(self, item):

        # select the ids of the current batch

        current_ids = self.data_ids[item*self.batch_size:(item+1)*self.batch_size]

        X, y = self.__generate_batch(current_ids)

        return X, y

    

    # collect the preprocessed images and targets of one batch

    def __generate_batch(self, current_ids):

        X = np.empty((self.batch_size, *self.input_shape, 3))

        y = np.empty((self.batch_size, self.num_classes))

        for idx, ident in enumerate(current_ids):

            # Store sample

            #image = self.preprocessor.preprocess(ident) 

            image = preprocess(ident,self.prep)

            X[idx] = image

            # Store class

            y[idx] = self.__get_target(ident)

        return X, y

    

    # extract the targets of one image id:

    def __get_target(self, ident):

        targets = self.dataframe.loc[ident].values

        return targets
def turn_pred_to_dataframe(data_df, pred):

    df = pd.DataFrame(pred, columns=data_df.columns, index=data_df.index)

    df = df.stack().reset_index()

    df.loc[:, "ID"] = df.id.str.cat(df.subtype, sep="_")

    df = df.drop(["id", "subtype"], axis=1)

    df = df.rename({0: "Label"}, axis=1)

    return df
test_dataloader = DataLoader(testdf,32,shuffle=False,input_shape=(224,224),prep="SIGMOID")

resnet_cat = tf.keras.models.load_model('../input/fork-of-ich-training-metrics/RESNET_SIGMOID_200_15_cat_cross.model')

resnet_cat.summary()
test_pred = resnet_cat.predict(test_dataloader,verbose=1)
pred = test_pred[0:testdf.shape[0]]

pred_df = turn_pred_to_dataframe(testdf,pred)

pred_df.to_csv("reset_cat_pred.csv",index=False)
test_dataloader = DataLoader(testdf,32,shuffle=False,input_shape=(224,224),prep="SIGMOID")

resnet_mfl = tf.keras.models.load_model('../input/fork-of-ich-training-metrics/RESNET_SIGMOID_200_15_focal_loss.model')

resnet_mfl.summary()

test_pred = resnet_mfl.predict(test_dataloader,verbose=1)
pred = test_pred[0:testdf.shape[0]]

pred_df = turn_pred_to_dataframe(testdf,pred)

pred_df.to_csv("reset_mfl_pred.csv",index=False)