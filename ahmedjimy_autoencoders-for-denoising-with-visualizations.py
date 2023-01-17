# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import glob

from pathlib import Path



from PIL import Image

import cv2



# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"

from sklearn.model_selection import train_test_split



# TensorFlow ≥2.0 is required

import tensorflow as tf

from tensorflow import keras

assert tf.__version__ >= "2.0"

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D,MaxPooling2D, BatchNormalization, UpSampling2D, Input, ZeroPadding2D, Cropping2D

from keras.preprocessing.image import load_img, array_to_img, img_to_array



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



# to make this notebook's output stable across runs

np.random.seed(42)

tf.random.set_seed(42)



# To plot pretty figures

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Define paths in the fancy way, after all we have pathlib now. No more os.path.join...whatever!!

input_dir  = Path('/kaggle/input/denoising-dirty-documents')

train = input_dir / 'train.zip'

train_cleaned = input_dir / 'train_cleaned.zip'

test = input_dir / 'test.zip'
# importing required modules 

from zipfile import ZipFile 



def extract_zip_file(file_names):

    # opening the zip file in READ mode 

    for file in file_names:

        with ZipFile(file, 'r') as zip: 

            zip.extractall() 



# train and test are zipped folders. we need to unzip, save them for further processing

extract_zip_file([train, train_cleaned, test])
# save directories of each image in a list using glob library

train_images = sorted(glob.glob('train/*.png'))

train_cleaned_images = sorted(glob.glob('train_cleaned/*.png'))

test_images = sorted(glob.glob('test/*.png'))
#  convert images to arrays for training

def convert_imgs_to_array(images_folder, test=False):

    images = []

    for img_dir in images_folder:

        # use keras built-in libraries load_img and im_to_array

        image = load_img(img_dir, color_mode='grayscale', target_size=(258, 540, 1))

        image = img_to_array(image).astype('float32') / 255.0

        images.append(image)

    return np.asarray(images)



X_train_full = convert_imgs_to_array(train_images)

Y_train_full = convert_imgs_to_array(train_cleaned_images)

#X_test = convert_imgs_to_array(test_images)
print(X_train_full.shape)

print(Y_train_full.shape)

#print(X_test.shape)
#split into training and validation

X_train, x_val, Y_train, y_val = train_test_split(X_train_full, Y_train_full,

                                                  test_size=0.3, random_state=42)
print(X_train.shape)

print(x_val.shape)

print(Y_train.shape)

print(y_val.shape)

#print(X_test.shape)
# plot random documents from training samples and their labels

def plot_documents(nrows=3, ncols=2):

    selection = np.random.choice(len(X_train), size=(nrows*ncols), replace=False)

    images = np.asarray(train_images)[selection]

    cleaned_images = np.asarray(train_cleaned_images)[selection]

    fig, axes = plt.subplots(figsize=(nrows*20, ncols*30), nrows=nrows, ncols=ncols)

    fig.subplots_adjust(hspace = .05, wspace=.05)

    axes = axes.ravel()

    for img, img_cleaned, i in zip(images, cleaned_images, range(0, nrows*ncols, 2)):

        axes[i].imshow(cv2.imread(img, cv2.IMREAD_GRAYSCALE), cmap='gray')

        axes[i+1].imshow(cv2.imread(img_cleaned, cv2.IMREAD_GRAYSCALE), cmap='gray')

        axes[i].axis('off')

        axes[i+1].axis('off')

        

plot_documents()
# try that later

'''

data_augmentation = keras.preprocessing.image.ImageDataGenerator(

        rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.05)  # randomly shift images vertically (fraction of total height)



data_augmentation.fit(X_train)



# plot random samples with keypoints from training

def plot_documents_after_augmentation(nrows=4, ncols=2):

    for X_batch, _ in data_augmentation.flow(X_train, Y_train, batch_size=nrows*ncols):

        print(X_batch.shape)

        fig, axes = plt.subplots(figsize=(nrows*20, ncols*30), nrows=nrows, ncols=ncols)

        fig.subplots_adjust(hspace=.05, wspace=.05)

        

        # create a grid of 3x3 images

        for i, axes in zip(range(0, nrows*ncols), axes.ravel()):

            axes.imshow(X_batch[i].reshape(258, 540), cmap='gray')

        break

   # for img, img_cleaned, i in zip(images, cleaned_images, range(0, nrows*ncols, 2)):

    #    axes[i].imshow(cv2.imread(img, cv2.IMREAD_GRAYSCALE), cmap='gray')

     #   axes[i+1].imshow(cv2.imread(img_cleaned, cv2.IMREAD_GRAYSCALE), cmap='gray')

      #  axes[i].axis('off')

       # axes[i+1].axis('off')

        

plot_documents_after_augmentation()

'''
'''

# CNN model architecture (all these parameters might be tuned to achieve better results)

from functools import partial



DefaultConv2D = partial(Conv2D, activation='relu', padding='SAME')



model = Sequential([

    # encoder

    DefaultConv2D(filters=64, kernel_size=3, input_shape=[258,540,1]),

    BatchNormalization(),

    MaxPooling2D((2, 2)),

    DefaultConv2D(filters=32, kernel_size=3),

    BatchNormalization(),

    MaxPooling2D((2, 2)),

    # decoder

    DefaultConv2D(filters=32, kernel_size=3),

    BatchNormalization(),

    UpSampling2D((2,2)),

    DefaultConv2D(filters=64, kernel_size=3),

    BatchNormalization(),

    UpSampling2D((2,2)),

    DefaultConv2D(filters=1, kernel_size=3, activation='sigmoid'),

    ZeroPadding2D((1,0))

])



# show model architecture

model.summary()

'''



'''

# CNN model architecture (all these parameters might be tuned to achieve better results)

from functools import partial



DefaultConv2D = partial(Conv2D, activation='relu', padding='SAME')



model = Sequential([

    # input layer

    BatchNormalization(input_shape=(258, 540, 1)),

    DefaultConv2D(64, (5, 5), kernel_initializer='he_normal'),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    # layer 2

    DefaultConv2D(32, (5, 5)),

    MaxPooling2D(pool_size=(2, 2)),

    Dropout(0.2),

    # layer 3

    DefaultConv2D(16, (5, 5)),

    UpSampling2D((2, 2)),

    Dropout(0.2),

    # layer 4

    DefaultConv2D(32, (3, 3)),

    UpSampling2D((2, 2)),

    Dropout(0.2),

    # layer 5

    DefaultConv2D(64, (3, 3)),

    Dropout(0.2),

    DefaultConv2D(1, (3, 3)),

    ZeroPadding2D((1,0))

])



model.summary()

'''
# Lets' define our autoencoder now

def build_autoenocder():

    input_img = Input(shape=(None,None,1), name='image_input')

    

    #enoder 

    x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv1')(input_img)

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv2')(x)

    x = MaxPooling2D((2,2), padding='same', name='pool1')(x)

    #x = MaxPooling2D((2,2), padding='same', name='pool2')(x)

    

    #decoder

    x = Conv2D(64, (3,3), activation='relu', padding='same', name='Conv3')(x)

    x = Conv2D(32, (3,3), activation='relu', padding='same', name='Conv4')(x)

    x = UpSampling2D((2,2), name='upsample2')(x)

    x = Conv2D(1, (3,3), activation='sigmoid', padding='same', name='Conv5')(x)

    

    #model

    autoencoder = Model(inputs=input_img, outputs=x)

    return autoencoder



model = build_autoenocder()

model.summary()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# another method is to use LearningRateScheduler, reduce the learning rate by 10% every epoch

# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)

K = keras.callbacks

reduce_lr = K.ReduceLROnPlateau(monitor='val_mae', patience=7,

                                             verbose=1, factor=0.1, min_lr=0.00001)



early_stopping = K.EarlyStopping(monitor='val_mae', patience=20, restore_best_weights=True,

                                 verbose=1, mode='auto')

#checkpointer = K.ModelCheckpoint(filepath = 'best_model.hdf5', monitor='val_mae',

                                 #verbose=1, save_weights_only=True)
epochs = 600

batch_size = 2

'''

history = model.fit_generator(data_augmentation.flow(X_train, Y_train, batch_size=batch_size),

                              epochs=epochs, validation_data=(x_val, y_val), 

                              steps_per_epoch=X_train.shape[0] // batch_size,

                              shuffle=True, verbose=1,

                              callbacks=[reduce_lr, early_stopping])

'''

history = model.fit(X_train, Y_train, validation_data=(x_val, y_val),

                   batch_size=batch_size, epochs=epochs, shuffle=True,

                   callbacks=[reduce_lr])
final_loss, final_mae = model.evaluate(x_val, y_val, verbose=1)

print("Final loss: {0:.4f}, final mae: {1:.4f}".format(final_loss, final_mae))
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['mae'], color='b', label="Training mae")

ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")

legend = ax[1].legend(loc='best', shadow=True)
# plot random samples with keypoints from training

def plot_documents_after_training(nrows=3, ncols=2):

    selection = np.random.choice(len(test_images), size=(nrows*ncols), replace=False)

    images = np.asarray(test_images)[selection]

    #tested_images = X_test[selection]

    #predicted_images = model.predict(tested_images)

    fig, axes = plt.subplots(figsize=(nrows*20, ncols*30), nrows=nrows, ncols=ncols)

    fig.subplots_adjust(hspace=.05, wspace=.05)

    axes = axes.ravel()

    for img, i in zip(images, range(0, nrows*ncols, 2)):

        original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        img = load_img(img, color_mode='grayscale', target_size=(original_img.shape[0], original_img.shape[1], 1))

        img = img_to_array(img).astype('float32') / 255.0

        predicted_img = model.predict(np.expand_dims(img, axis=0))

        axes[i].imshow(array_to_img(img), cmap='gray')

        axes[i+1].imshow(array_to_img(predicted_img[0]), cmap='gray')

        axes[i].axis('off')

        axes[i+1].axis('off')

        

plot_documents_after_training()
# save the weights to prevent training every time you open the kernel

model.save_weights("model.h5")
# after loading, you have to compile the model

model.load_weights('/kaggle/input/denoising-dirty-documents2/model.h5')

model.compile(optimizer='adam', loss='mse',

             metrics=['mae'])



final_loss, final_mae = model.evaluate(x_val, y_val, verbose=0)

print("Final loss: {0:.4f}, final mae: {1:.4f}".format(final_loss, final_mae))
def predict_images(test_images):

    results = []

    for img in test_images:

        original_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

        img = load_img(img, color_mode='grayscale', target_size=(original_img.shape[0], original_img.shape[1], 1))

        img = img_to_array(img).astype('float32') / 255.0

        predicted_img = model.predict(np.expand_dims(img, axis=0))

        results.append(predicted_img)

    return results



results = predict_images(test_images)
print(len(results))

print(results[2].shape)
ids = []

values = []



# extract image ids from image file names ('train/509.png --> '509')

def split_numbers(s):

    head = s.split('.')[0].split('/')[1]

    return head



for i, pred in enumerate(results):

    print('Predicted image shape: ', pred.shape)

    img_dir = test_images[i]

    img_id = split_numbers(img_dir)   



    for j in range(pred.shape[2]):

        for k in range(pred.shape[1]):

            values.append(pred[0][k][j].item())

            ids.append(img_id + '_' + str(k+1) + '_' + str(j+1))

    print("Processed: {}".format(img_id))   
len(values)
pd.DataFrame({'id': ids, 'value': values}).to_csv('submission.csv', index=False)
my_submission = pd.read_csv("submission.csv")
my_submission.head()