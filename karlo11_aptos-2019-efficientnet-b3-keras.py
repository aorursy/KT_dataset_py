import json

import math

import os

import sys

import cv2

import numpy as np

import keras

from keras import layers

from keras import applications

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score,confusion_matrix, f1_score, precision_score, recall_score

import scipy as sp

#import tensorflow as tf

from tqdm import tqdm

from functools import partial
sys.path.append(os.path.abspath('../input/efficientnet/efficientnet-master/efficientnet-master/'))

from keras import applications

from efficientnet import EfficientNetB3

np.random.seed(2019)

#tf.set_random_seed(2019)



train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print(train_df.shape)

print(test_df.shape)

train_df.head()



train_df['diagnosis'].value_counts().sort_index().plot(kind="bar", 

                                                       figsize=(12,5), 

                                                       rot=0)

plt.xlabel("Label", fontsize=15)

plt.ylabel("Frequency", fontsize=15)
def crop_image_from_gray(img, tol=7):

    """

    Applies masks to the orignal image and 

    returns the a preprocessed image with 

    3 channels

    """

    # If for some reason we only have two channels

    if img.ndim == 2:

        mask = img > tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    # If we have a normal RGB images

    elif img.ndim == 3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img > tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

            img = np.stack([img1,img2,img3],axis=-1)

        return img



def preprocess_image(path, sigmaX=10):

    """

    The whole preprocessing pipeline:

    1. Read in image

    2. Apply masks

    3. Resize image to desired size

    4. Add Gaussian noise to increase Robustness

    """

    image = cv2.imread(path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_image_from_gray(image)

    image = cv2.resize(image, (300, 300))

    image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 101)

    return image
fig, ax = plt.subplots(1, 5, figsize=(15, 6))

for i in range(5):

    sample = train_df[train_df['diagnosis'] == i].sample(1)

    image_name = sample['id_code'].item()

    X = preprocess_image(f"../input/aptos2019-blindness-detection/train_images/{image_name}.png")

    ax[i].set_title(f"Image: {image_name}\n Label = {sample['diagnosis'].item()}", 

                    weight='bold', fontsize=10)

    ax[i].axis('off')

    ax[i].imshow(X);

N = train_df.shape[0]

x_train = np.empty((N, 300, 300, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(train_df['id_code'])):

    x_train[i, :, :, :] = preprocess_image(

         f'../input/aptos2019-blindness-detection/train_images/{image_id}.png'

     )

N = test_df.shape[0]

x_test = np.empty((N, 300, 300, 3), dtype=np.uint8)



for i, image_id in enumerate(tqdm(test_df['id_code'])):

    x_test[i, :, :, :] = preprocess_image(

        f'../input/aptos2019-blindness-detection/test_images/{image_id}.png'

    )

y_train = pd.get_dummies(train_df['diagnosis']).values



print(train_df.shape)

print(test_df.shape)#error



y_train_multi = np.empty(y_train.shape, dtype=y_train.dtype)

y_train_multi[:, 4] = y_train[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))
x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=0.15, 

    random_state=2019

)

BATCH_SIZE = 32



def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.02,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,

    )



# Using original generator

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=2019)

class Metrics(Callback):

    def on_train_begin(self, logs={}):

        self.val_kappas = []



    def on_epoch_end(self, epoch, logs={}):

        X_val, y_val = self.validation_data[:2]

        y_val = y_val.sum(axis=1) - 1

        

        y_pred = self.model.predict(X_val) > 0.5

        y_pred = y_pred.astype(int).sum(axis=1) - 1



        _val_kappa = cohen_kappa_score(

            y_val,

            y_pred, 

            weights='quadratic'

        )



        self.val_kappas.append(_val_kappa)



        print(f"val_kappa: {_val_kappa:.4f}")

        

        if _val_kappa == max(self.val_kappas):

            print("Validation Kappa has improved. Saving model.")

            self.model.save('model.h5')



        return



kappa_metrics = Metrics()
effnet = EfficientNetB3(weights=None,

                        include_top=False,

                        input_shape=(300, 300, 3))

effnet.load_weights('../input/efficientnet-keras-weights-b0b5/efficientnet-b3_imagenet_1000_notop.h5')



def build_model():



    model = Sequential()

    model.add(effnet)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(5, activation='sigmoid'))



    model.compile(

        loss='binary_crossentropy',

        optimizer='adam',

        metrics=['accuracy'])

    return model

model = build_model()

model.summary()



history = model.fit_generator(data_generator, 

                              steps_per_epoch=100, 

                              epochs=20, validation_data=(x_val, y_val), 

                              callbacks=[kappa_metrics]

                             )

with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
model.load_weights('model.h5')

y_val_pred = model.predict(x_val)



y_test = model.predict(x_test) > 0.5

y_test = y_test.astype(int).sum(axis=1) - 1



test_df['diagnosis'] = y_test

test_df.to_csv('submission.csv',index=False)



test_df['diagnosis'].value_counts().sort_index().plot(kind="bar", 

                                                       figsize=(12,5), 

                                                       rot=0)

plt.xlabel("Label", fontsize=15)

plt.ylabel("Frequency", fontsize=15)





                                       