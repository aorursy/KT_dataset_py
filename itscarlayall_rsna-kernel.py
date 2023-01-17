import matplotlib.pyplot as plt

import pandas as pd

import pydicom

import numpy as np

import os

from os import listdir

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50

from keras.engine.input_layer import Input

from keras import layers

from keras.models import Model, Sequential

from keras.optimizers import Adam

from keras.initializers import he_normal

from keras.regularizers import l2

from keras.callbacks import callbacks

import tensorflow as tf

from tensorflow.compat.v2.math import log

import csv

from sklearn.model_selection import train_test_split

import cv2
TRAIN_IMG_PATH = "../input/my_train_sample/"

TEST_IMG_PATH = "../input/stage_1_test_images_sample/"

TRAIN_CSV_PATH = "../input/stage_1_train.csv"

TEST_CSV_PATH = "../input/stage_1_sample_submission.csv"
def format_train_csv(df_train):

    '''Pivot the columns to form a multilabel dataset'''

    ids = df_train["ID"].copy()

    df_train["Filename"] = ids.apply(lambda one_id: one_id[0:12] + ".dcm")

    df_train["Type"] = ids.apply(lambda one_id: one_id[13:len(one_id)])

    df_train = df_train[['Filename', 'Type', 'Label']]

    df_train.drop_duplicates(inplace=True)

    df_train = df_train.pivot(index='Filename', columns = 'Type', values = 'Label')

    return df_train

    
df_train = pd.read_csv(TRAIN_CSV_PATH, delimiter=",")

df_test = pd.read_csv(TEST_CSV_PATH, delimiter=",")



df_train = format_train_csv(df_train)

df_train
def rescale_pixelarray(dataset):

    '''Bound values that are too far below the range'''

    image = dataset.pixel_array

    rescaled_image = image * dataset.RescaleSlope + dataset.RescaleIntercept

    rescaled_image[rescaled_image < -1024] = -1024

    return rescaled_image



def window_image(img, window_center, window_width, rescale=True):

    '''Get a window from the dicom image'''

    img_min = window_center - window_width//2

    img_max = window_center + window_width//2

    img[img<img_min] = img_min

    img[img>img_max] = img_max

    

    if rescale:

        # Extra rescaling to 0-1, not in the original notebook

        img = (img - img_min) / (img_max - img_min)

    

    return img
RESHAPE_SIZE = 256



def create_train_dataset(window_center=75, window_width=210, reshape_size=RESHAPE_SIZE):

    '''Reads the dicom images, processes them and splits into training and validation sets'''

    # read train images and reformat them

    filenames_train = listdir(TRAIN_IMG_PATH)

    labels = df_train.loc[filenames_train]

    data = []



    for file in filenames_train:

        image = pydicom.dcmread(TRAIN_IMG_PATH + os.path.sep + file)

        image = rescale_pixelarray(image)

        image = window_image(image, window_center, window_width)

        image = cv2.resize(image, (RESHAPE_SIZE, RESHAPE_SIZE))

        image = np.stack((image,)*3, axis=-1)

        data.append(image)



    # get a validation set

    (train_x, val_x, train_y, val_y) = train_test_split(data, labels,

        test_size=0.25, random_state=42)



    train_x = np.asarray(train_x)

    train_y = np.asarray(train_y)

    val_x = np.asarray(train_x)

    val_y = np.asarray(train_y)

    

    return train_x, train_y, val_x, val_y

    

    

def create_test_dataset(window_center=75, window_width=210, reshape_size=RESHAPE_SIZE):

    '''Reads the dicom test images and processes them'''

    # read test images and reformat them

    filenames_test = listdir(TEST_IMG_PATH)

    test_x = []



    for file in filenames_test:

        image = pydicom.dcmread(TEST_IMG_PATH + os.path.sep + file)

        image = rescale_pixelarray(image)

        image = window_image(image, window_center, window_width)

        image = cv2.resize(image, (RESHAPE_SIZE, RESHAPE_SIZE))

        image = np.stack((image,)*3, axis=-1)

        test_x.append(image)



    return test_x



    
train_x, train_y, val_x, val_y = create_train_dataset()

test_x = create_test_dataset()
plt.imshow(train_x[10], cmap=plt.cm.bone)
def np_multilabel_loss(y_true, y_pred, class_weights=None):

    '''Definition of the multilabel loss cost function'''

    y_pred = np.where(y_pred > 1-(1e-07), 1-1e-07, y_pred)

    y_pred = np.where(y_pred < 1e-07, 1e-07, y_pred)

    single_class_cross_entropies = - np.mean(y_true * np.log(y_pred) + (1-y_true) * np.log(1-y_pred), axis=0)

    

    print(single_class_cross_entropies)

    if class_weights is None:

        loss = np.mean(single_class_cross_entropies)

    else:

        loss = np.sum(class_weights*single_class_cross_entropies)

    return loss



def get_raw_xentropies(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)

    xentropies = y_true * log(y_pred) + (1-y_true) * log(1-y_pred)

    return -xentropies



def multilabel_focal_loss(class_weights=None, alpha=0.5, gamma=2):

    '''Multilabel focal loss for unbalanced datasets'''

    def multilabel_focal_loss_inner(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        y_pred = tf.cast(y_pred, tf.float32)

        

        xentropies = get_raw_xentropies(y_true, y_pred)



        # compute pred_t:

        y_t = tf.where(tf.equal(y_true,1), y_pred, 1.-y_pred)

        alpha_t = tf.where(tf.equal(y_true, 1), alpha * tf.ones_like(y_true), (1-alpha) * tf.ones_like(y_true))



        # compute focal loss contributions

        focal_loss_contributions =  tf.multiply(tf.multiply(tf.pow(1-y_t, gamma), xentropies), alpha_t) 



        # our focal loss contributions have shape (n_samples, s_classes), we need to reduce with mean over samples:

        focal_loss_per_class = tf.reduce_mean(focal_loss_contributions, axis=0)



        # compute the overall loss if class weights are None (equally weighted):

        if class_weights is None:

            focal_loss_result = tf.reduce_mean(focal_loss_per_class)

        else:

            # weight the single class losses and compute the overall loss

            weights = tf.constant(class_weights, dtype=tf.float32)

            focal_loss_result = tf.reduce_sum(tf.multiply(weights, focal_loss_per_class))

            

        return focal_loss_result

    return multilabel_focal_loss_inner
def resnet_50():

    '''Sample ResNet50'''

    net = ResNet50(include_top=False, 

                   weights="imagenet", 

                   input_shape=(RESHAPE_SIZE, RESHAPE_SIZE, 3))

    for layer in net.layers:

        layer.trainable = False

        

    x = net.output

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    x = layers.Dense(100, activation="relu")(x)

    x = layers.Dropout(0.3)(x)

    pred = layers.Dense(6,

                 kernel_initializer=he_normal(seed=11),

                 kernel_regularizer=l2(0.05),

                 bias_regularizer=l2(0.05), activation="sigmoid")(x)

    model = Model(inputs=net.input, outputs=pred)

    

    model.compile(optimizer=Adam(lr=0.001),

                           loss='categorical_crossentropy', 

                           metrics=[multilabel_focal_loss(alpha=0.5, gamma=0)])

    return model
model = resnet_50()



# construct the training image generator for data augmentation

datagen = ImageDataGenerator()

EPOCHS = 20

BS = 64



# Define a checkpoint and early stopping

checkpoint = callbacks.ModelCheckpoint(filepath="resnet_50_best.h5",

                                       monitor="val_multilabel_focal_loss_inner",

                                       mode="min",

                                       verbose=1,

                                       save_best_only=True,

                                       save_weights_only=True,

                                       period=1)



e_stopping = callbacks.EarlyStopping(monitor="val_multilabel_focal_loss_inner",

                                min_delta=0.01,

                                patience=3,

                                mode="min",

                                restore_best_weights=True)



# train the network

history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=BS),

                              validation_data=(val_x, val_y), epochs=EPOCHS, 

                              callbacks=[checkpoint, e_stopping])

# plot multilabel focal loss

plt.plot(history.history['multilabel_focal_loss_inner'])

plt.plot(history.history['val_multilabel_focal_loss_inner'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()