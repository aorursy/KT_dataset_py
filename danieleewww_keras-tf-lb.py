# Random initialization

import numpy as np

np.random.seed(98643)

import tensorflow as tf

tf.set_random_seed(683)

# Uncomment this to hide TF warnings about allocation

#import os

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



# An image clearing dependencies

from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,

                                 denoise_wavelet, estimate_sigma, denoise_tv_bregman, denoise_nl_means)

from skimage.filters import gaussian

from skimage.color import rgb2gray



# Data reading and visualization

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import MinMaxScaler



# Training part

from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalAveragePooling2D, Lambda

from keras.layers import GlobalMaxPooling2D

from keras.layers.normalization import BatchNormalization

from keras.layers.merge import Concatenate

from keras.models import Model

from keras.optimizers import Adam, SGD

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from sklearn.model_selection import train_test_split

from sklearn.utils import shuffle



# Any results you write to the current directory are saved as output.
# Translate data to an image format

def color_composite(data):

    rgb_arrays = []

    for i, row in data.iterrows():

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 / band_2



        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))

        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))

        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))



        rgb = np.dstack((r, g, b))

        rgb_arrays.append(rgb)

    return np.array(rgb_arrays)



def denoise(X, weight, multichannel):

    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel) for item in X])



def smooth(X, sigma):

    return np.asarray([gaussian(item, sigma=sigma) for item in X])



def grayscale(X):

    return np.asarray([rgb2gray(item) for item in X])
train = pd.read_json("../input/train.json")

train.inc_angle = train.inc_angle.replace('na', 0)

train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

train_all = True



# These are train flags that required to train model more efficiently and 

# select proper model parameters

train_b = True or train_all

train_img = True or train_all

train_total = True or train_all

predict_submission = True and train_all



clean_all = True

clean_b = True or clean_all

clean_img = True or clean_all



load_all = False

load_b = False or load_all

load_img = False or load_all
def create_dataset(frame, labeled, smooth_rgb=0.2, smooth_gray=0.5,

                   weight_rgb=0.05, weight_gray=0.05):

    band_1, band_2, images = frame['band_1'].values, frame['band_2'].values, color_composite(frame)

    to_arr = lambda x: np.asarray([np.asarray(item) for item in x])

    band_1 = to_arr(band_1)

    band_2 = to_arr(band_2)

    band_3 = (band_1 + band_2) / 2

    gray_reshape = lambda x: np.asarray([item.reshape(75, 75) for item in x])

    # Make a picture format from flat vector

    band_1 = gray_reshape(band_1)

    band_2 = gray_reshape(band_2)

    band_3 = gray_reshape(band_3)

    print('Denoising and reshaping')

    if train_b and clean_b:

        # Smooth and denoise data

        band_1 = smooth(denoise(band_1, weight_gray, False), smooth_gray)

        print('Gray 1 done')

        band_2 = smooth(denoise(band_2, weight_gray, False), smooth_gray)

        print('Gray 2 done')

        band_3 = smooth(denoise(band_3, weight_gray, False), smooth_gray)

        print('Gray 3 done')

    if train_img and clean_img:

        images = smooth(denoise(images, weight_rgb, True), smooth_rgb)

    print('RGB done')

    tf_reshape = lambda x: np.asarray([item.reshape(75, 75, 1) for item in x])

    band_1 = tf_reshape(band_1)

    band_2 = tf_reshape(band_2)

    band_3 = tf_reshape(band_3)

    #images = tf_reshape(images)

    band = np.concatenate([band_1, band_2, band_3], axis=3)

    X_angle = np.array(frame.inc_angle)

    if labeled:

        y = np.array(frame["is_iceberg"])

    else:

        y = None

    return y, X_angle, band, images
y_train, X_angles, X_b, X_images = create_dataset(train, True)
fig = plt.figure(200, figsize=(15, 15))

random_indicies = np.random.choice(range(len(X_images)), 9, False)

subset = X_images[random_indicies]

for i in range(9):

    ax = fig.add_subplot(3, 3, i + 1)

    ax.imshow(subset[i])

plt.show()
fig = plt.figure(202, figsize=(15, 15))

band_1_x = train['band_1'].values

subset = np.asarray(band_1_x)[random_indicies]

subset = np.asarray([np.asarray(item).reshape(75, 75) for item in subset])

for i in range(9):

    ax = fig.add_subplot(3, 3, i + 1)

    ax.imshow(subset[i])

plt.show()
fig = plt.figure(202, figsize=(15, 15))

subset = np.asarray(band_1_x)[random_indicies]

subset = denoise(np.asarray([np.asarray(item).reshape(75, 75) for item in subset]), 0.05, False)

for i in range(9):

    ax = fig.add_subplot(3, 3, i + 1)

    ax.imshow(subset[i])

plt.show()
fig = plt.figure(202, figsize=(15, 15))

subset = np.asarray(band_1_x)[random_indicies]

subset = smooth(denoise(np.asarray(

    [np.asarray(item).reshape(75, 75) for item in subset]), 0.05, False), 0.5)

for i in range(9):

    ax = fig.add_subplot(3, 3, i + 1)

    ax.imshow(subset[i])

plt.show()
def get_model_notebook(angle, lr, decay, channels, relu_type='relu'):

    # angle variable defines if we should use angle parameter or ignore it

    input_1 = Input(shape=(75, 75, channels))

    input_2 = Input(shape=[1])



    fcnn = Conv2D(32, kernel_size=(3, 3), activation=relu_type)(input_1)

    fcnn = MaxPooling2D((3, 3))(fcnn)

    fcnn = Dropout(0.2)(fcnn)

    fcnn = Conv2D(64, kernel_size=(3, 3), activation=relu_type)(fcnn)

    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)

    fcnn = Dropout(0.2)(fcnn)

    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)

    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)

    fcnn = Dropout(0.2)(fcnn)

    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)

    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)

    fcnn = Dropout(0.2)(fcnn)

    fcnn = Flatten()(fcnn)

    if angle:

        local_input = [input_1, input_2]

    else:

        local_input = input_1

    dense = Dropout(0.2)(fcnn)

    dense = Dense(256, activation=relu_type)(dense)

    partial_model = Model(input_1, fcnn)

    dense = Dropout(0.2)(dense)

    dense = Dense(128, activation=relu_type)(dense)

    dense = Dropout(0.2)(dense)

    dense = Dense(64, activation=relu_type)(dense)

    dense = Dropout(0.2)(dense)

    # For some reason i've decided not to normalize angle data

    if angle:

        dense = Concatenate()([dense, input_2])

    else:

        dense = dense

    output = Dense(1, activation="sigmoid")(dense)

    model = Model(local_input, output)

    optimizer = Adam(lr=lr, decay=decay)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model, partial_model

def combined_model(m_b, m_img, lr, decay):

    input_b = Input(shape=(75, 75, 3))

    input_img = Input(shape=(75, 75, 3))

    input_angular = Input(shape=[1])



    # I've never tested non-trainable source models tho

    #for layer in m_b.layers:

    #    layer.trainable = False

    #for layer in m_img.layers:

    #    layer.trainable = False



    m1 = m_b(input_b)

    m2 = m_img(input_img)



    # So, combine models and train perceptron based on that

    # The iteresting idea is to use XGB for this task, but i actually hate this method

    common = Concatenate()([m1, m2])

    common = BatchNormalization()(common)

    common = Dropout(0.3)(common)

    common = Dense(2048, activation='relu')(common)

    common = Dropout(0.3)(common)

    common = Dense(1024, activation='relu')(common)

    common = Dropout(0.3)(common)

    common = Dense(512, activation='relu')(common)

    common = Dropout(0.3)(common)

    common = Concatenate()([common, BatchNormalization()(input_angular)])

    output = Dense(1, activation="sigmoid")(common)

    model = Model([input_b, input_img, input_angular], output)

   # optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)

    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model
def train_model(model, batch_size, epochs, checkpoint_name, X_train, y_train, verbose=2, val_data=None, val_split=0.15):

    callbacks = [ModelCheckpoint(checkpoint_name, save_best_only=True, monitor='val_loss')]

    try:

        if val_data is None:

            model.fit(X_train, y_train, epochs=epochs, validation_split=val_split,

                      batch_size=batch_size, callbacks=callbacks, verbose=verbose, shuffle=True)

        else:

            x_val, y_val = val_data

            model.fit(X_train, y_train, epochs=epochs, validation_data=[x_val, y_val],

                      batch_size=batch_size, callbacks=callbacks, verbose=verbose, shuffle=True)

    except KeyboardInterrupt:

        if verbose > 0:

            print('Interrupted')

    if verbose > 0:

        print('Loading model')

    model.load_weights(filepath=checkpoint_name)

    return model
def get_angular_status(angle, x, x_a):

    if angle:

        result = [x, x_a]

    else:

        result = x

    return result

#Train a particular model

def gen_model_weights(angle, lr, decay, channels, relu, batch_size, epochs, path_name, data, only_load=False, verbose=2):

    X_train, X_angle_train, y_train, X_val, X_angles_val, y_val = data

    X_train, X_angle_train, y_train = shuffle(X_train, X_angle_train, y_train, random_state=np.random.randint(1, 123))

    model, partial_model = get_model_notebook(angle, lr, decay, channels, relu)

    if only_load:

        model.load_weights(path_name)

        return model, partial_model

    model = train_model(model, batch_size, epochs, path_name,

                           get_angular_status(angle, X_train, X_angle_train), y_train, verbose=verbose)



    if verbose > 0:

        loss_val, acc_val = model.evaluate(get_angular_status(angle, X_val, X_angles_val), y_val,

                               verbose=0, batch_size=batch_size)



        loss_train, acc_train = model.evaluate(get_angular_status(angle, X_train, X_angle_train), y_train,

                                       verbose=0, batch_size=batch_size)



        print('Val/Train Loss:', str(loss_val) + '/' + str(loss_train), \

            'Val/Train Acc:', str(acc_val) + '/' + str(acc_train))

    return model, partial_model
# Train all 3 models

def train_models(dataset, lr, batch_size, max_epoch, verbose=2, return_model=False):

    X_angles, y_train, X_b, X_images = dataset

    angle_b = True

    angle_images = True

    X_angles, X_angles_val,\

    y_train, y_val,\

    X_b, X_b_val,\

    X_images, X_images_val = train_test_split(X_angles, y_train, X_b, X_images, random_state=687, train_size=0.9)



    if train_b:

        if verbose > 0:

            print('Training bandwidth network')

        data_b1 = (X_b, X_angles, y_train, X_b_val, X_angles_val, y_val)

        model_b, model_b_cut = gen_model_weights(angle_b, lr, 0, 3, 'relu', batch_size, max_epoch, 'model_b',

                                             data_b1, only_load=load_b, verbose=verbose)



    if train_img:

        if verbose > 0:

            print('Training image network')

        data_images = (X_images, X_angles, y_train, X_b_val, X_angles_val, y_val)

        model_images, model_images_cut = gen_model_weights(angle_images, lr, 0, 3, 'relu', batch_size, max_epoch, 'model_img',

                                                       data_images, only_load=load_img, verbose=verbose)



    if train_total:

        common_model = combined_model(model_b_cut, model_images_cut, lr, 0)

        common_x_train = [X_b, X_images, X_angles]

        common_y_train = y_train

        common_x_val = [X_b_val, X_images_val, X_angles_val]

        common_y_val = y_val

        if verbose > 0:

            print('Training common network')

        common_model = train_model(common_model, batch_size, max_epoch, 'common_check', common_x_train,

                           common_y_train, verbose=verbose, val_split=0.2)



        loss_val, acc_val = common_model.evaluate(common_x_val, common_y_val,

                                           verbose=0, batch_size=batch_size)

        loss_train, acc_train = common_model.evaluate(common_x_train, common_y_train,

                                                  verbose=0, batch_size=batch_size)

        if verbose > 0:

            print('Loss:', loss_val, 'Acc:', acc_val)

    if return_model:

        return common_model

    else:

        return (loss_train, acc_train), (loss_val, acc_val)
# Best parameters i got are

# epochs : 250

# learning rate : 8e-5

# batch size : 32

# CARE: The image model is overfits with parameters used here

common_model = train_models((X_angles, y_train, X_b, X_images), 5e-04, 32, 50, 1, return_model=True)
if predict_submission:

    print('Reading test dataset')

    test = pd.read_json("../input/test.json")

    test.inc_angle = test.inc_angle.replace('na', 0)

    test.inc_angle = test.inc_angle.astype(float).fillna(0.0)

    y_fin, X_angle_fin, X_fin_b, X_fin_img = create_dataset(test, False)

    print('X shape:', X_fin_img.shape)

    print('X angle shape:', X_angle_fin.shape)

    print('Predicting')

    prediction = common_model.predict([X_fin_b, X_fin_img, X_angle_fin], verbose=1, batch_size=32)

    print('Submitting')

    submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})



    submission.to_csv("./submission.csv", index=False)

    print('Done')