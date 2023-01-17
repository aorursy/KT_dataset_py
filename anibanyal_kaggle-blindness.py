import pandas as pd

import os, shutil

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import cv2

from keras import models, layers

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg19 import VGG19
base_dir = '/kaggle/input/aptos2019-blindness-detection'

# for f in os.listdir(base_dir):

#     print(f)
csv_path = os.path.join(base_dir, 'train.csv')

df = pd.read_csv(csv_path)

df = df.sort_values(by='id_code')

df['id_code'] = df['id_code'].values + '.png' ## needed for flow_from_dataframe

df.head()
labels, counts = np.unique(df['diagnosis'], return_counts=True)

# plt.bar(labels, counts)

# plt.title('Bar chart of labels')

# plt.show()
TRAIN_IMG_PATH = os.path.join(base_dir, 'train_images')

IMG_WIDTH = 256

IMG_HEIGHT = 256
# def crop_image_from_gray(img, tol=7):

#     """

#     Applies masks to the orignal image and 

#     returns the a preprocessed image with 

#     3 channels

    

#     :param img: A NumPy Array that will be cropped

#     :param tol: The tolerance used for masking

    

#     :return: A NumPy array containing the cropped image

#     """

#     # If for some reason we only have two channels

#     if img.ndim == 2:

#         mask = img > tol

#         return img[np.ix_(mask.any(1),mask.any(0))]

#     # If we have a normal RGB images

#     elif img.ndim == 3:

#         gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#         mask = gray_img > tol

        

#         check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

#         if (check_shape == 0): # image is too dark so that we crop out everything,

#             return img # return original image

#         else:

#             img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

#             img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

#             img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

#             img = np.stack([img1,img2,img3],axis=-1)

#         return img



def preprocess_image(image, sigmaX=10):

    """

    The whole preprocessing pipeline:

    1. Read in image

    2. Apply masks

    3. Resize image to desired size

    4. Add Gaussian noise to increase Robustness

    

    :param img: A NumPy Array that will be cropped

    :param sigmaX: Value used for add GaussianBlur to the image

    

    :return: A NumPy array containing the preprocessed image

    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     image = crop_image_from_gray(image)

    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    return image
train_datagen = ImageDataGenerator(rescale=1/128., horizontal_flip=True,

                                   vertical_flip=True, validation_split=0.1, preprocessing_function=preprocess_image)

train_generator = train_datagen.flow_from_dataframe(df, x_col='id_code', y_col='diagnosis'

                                                    , directory=TRAIN_IMG_PATH, class_mode='raw' )
conv_base = VGG19(weights='imagenet',

                        include_top=False,

                        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

conv_base.trainable = False
model = models.Sequential([

    conv_base,

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),

    layers.Dense(1, activation='linear')

])

model.compile(loss='mse', optimizer='adam', metrics=['mse', 'acc'])

BATCH_SIZE = 32
history = model.fit_generator(train_generator, steps_per_epoch=train_generator.samples // BATCH_SIZE,

                              epochs=30, verbose=True)