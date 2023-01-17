# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import math

import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.models import Sequential, Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from albumentations.core.transforms_interface import ImageOnlyTransform

from tensorflow.keras import layers, optimizers

from sklearn.model_selection import train_test_split

import cv2

from albumentations.augmentations import functional as F

import random

import albumentations

from keras.utils import plot_model

from PIL import Image

from PIL import ImageOps

from keras.callbacks import (Callback, ModelCheckpoint,

                                        LearningRateScheduler,EarlyStopping, 

                                        ReduceLROnPlateau,CSVLogger)



# Any results you write to the current directory are saved as output.
#We already went through spiritual and holistic EDA in many kernels including mine, let's get started with network
PATH='../input/bengaliai-cv19/'

SEED=2019

HEIGHT=137

WIDTH=236

batch_size=64



resizedimgs='../input/grapheme-imgs-128x128/'

mobilenetpath='../input/mobilenet-v2-128'



class_map = pd.read_csv(PATH+"class_map.csv")

sample_submission = pd.read_csv(PATH+"sample_submission.csv")

test = pd.read_csv(PATH+"test.csv")

train = pd.read_csv(PATH+"train.csv")



train['filename'] = train.image_id.apply(lambda filename: resizedimgs + filename + '.png')



def seed_all(SEED):

    random.seed(SEED)

    np.random.seed(SEED)

    

# seed all

seed_all(SEED)

train.head()
train_files, valid_files, y_train, y_valid = train_test_split(

    train.filename.values, 

    train[['grapheme_root','vowel_diacritic', 'consonant_diacritic']].values, 

    test_size=0.25, 

    random_state=2019

)
def int_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval .

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    An int that results from scaling `maxval` according to `level`.

    """

    return int(level * maxval / 10)





def float_parameter(level, maxval):

    """Helper function to scale `val` between 0 and maxval.

    Args:

    level: Level of the operation that will be between [0, `PARAMETER_MAX`].

    maxval: Maximum value that the operation can have. This will be scaled to

      level/PARAMETER_MAX.

    Returns:

    A float that results from scaling `maxval` according to `level`.

    """

    return float(level) * maxval / 10.





def sample_level(n):

    return np.random.uniform(low=0.1, high=n)





def autocontrast(pil_img, _):

    return ImageOps.autocontrast(pil_img)





def equalize(pil_img, _):

    return ImageOps.equalize(pil_img)





def posterize(pil_img, level):

    level = int_parameter(sample_level(level), 4)

    return ImageOps.posterize(pil_img, 4 - level)





def rotate(pil_img, level):

    degrees = int_parameter(sample_level(level), 30)

    if np.random.uniform() > 0.5:

        degrees = -degrees

    return pil_img.rotate(degrees, resample=Image.BILINEAR)





def solarize(pil_img, level):

    level = int_parameter(sample_level(level), 256)

    return ImageOps.solarize(pil_img, 256 - level)





def shear_x(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, level, 0, 0, 1, 0),

                           resample=Image.BILINEAR)





def shear_y(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, 0, level, 1, 0),

                           resample=Image.BILINEAR)





def translate_x(pil_img, level):

    level = int_parameter(sample_level(level), pil_img.size[0] / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, level, 0, 1, 0),

                           resample=Image.BILINEAR)





def translate_y(pil_img, level):

    level = int_parameter(sample_level(level), pil_img.size[0] / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform(pil_img.size,

                           Image.AFFINE, (1, 0, 0, 0, 1, level),

                           resample=Image.BILINEAR)





# operation that overlaps with ImageNet-C's test set

def color(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Color(pil_img).enhance(level)





# operation that overlaps with ImageNet-C's test set

def contrast(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Contrast(pil_img).enhance(level)





# operation that overlaps with ImageNet-C's test set

def brightness(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Brightness(pil_img).enhance(level)





# operation that overlaps with ImageNet-C's test set

def sharpness(pil_img, level):

    level = float_parameter(sample_level(level), 1.8) + 0.1

    return ImageEnhance.Sharpness(pil_img).enhance(level)





augmentations = [

    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,

    translate_x, translate_y

]



augmentations_all = [

    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,

    translate_x, translate_y, color, contrast, brightness, sharpness

]



def normalize(image):

    """Normalize input image channel-wise to zero mean and unit variance."""

    return image - 127



def apply_op(image, op, severity):

    #   image = np.clip(image, 0, 255)

    pil_img = Image.fromarray(image)  # Convert to PIL.Image

    pil_img = op(pil_img, severity)

    return np.asarray(pil_img)



def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):

    """Perform AugMix augmentations and compute mixture.

    Args:

    image: Raw input image as float32 np.ndarray of shape (h, w, c)

    severity: Severity of underlying augmentation operators (between 1 to 10).

    width: Width of augmentation chain

    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly

      from [1, 3]

    alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:

    mixed: Augmented and mixed image.

    """

    ws = np.float32(

      np.random.dirichlet([alpha] * width))

    m = np.float32(np.random.beta(alpha, alpha))



    mix = np.zeros_like(image).astype(np.float32)

    for i in range(width):

        image_aug = image.copy()

        depth = depth if depth > 0 else np.random.randint(1, 4)

        for _ in range(depth):

            op = np.random.choice(augmentations)

            image_aug = apply_op(image_aug, op, severity)

        # Preprocessing commutes since all coefficients are convex

        mix += ws[i] * image_aug

#         mix += ws[i] * normalize(image_aug)



    mixed = (1 - m) * image + m * mix

#     mixed = (1 - m) * normalize(image) + m * mix

    return mixed





class RandomAugMix(ImageOnlyTransform):



    def __init__(self, severity=3, width=3, depth=-1, alpha=1., always_apply=False, p=0.5):

        super().__init__(always_apply, p)

        self.severity = severity

        self.width = width

        self.depth = depth

        self.alpha = alpha



    def apply(self, image, **params):

        image = augment_and_mix(

            image,

            self.severity,

            self.width,

            self.depth,

            self.alpha

        )

        return image

    
def data_generator(filenames, y,type='train', batch_size=64, shape=(128, 128, 1), random_state=2019,transform=None):

    

    indices = np.arange(len(filenames))

    

    while True:

        np.random.shuffle(indices)

        

        for i in range(0, len(indices), batch_size):

            batch_idx = indices[i:i+batch_size]

            size = len(batch_idx)

            

            batch_files = filenames[batch_idx]

            X_batch = np.zeros((size, *shape))

            y_batch = y[batch_idx]

            

            for i, file in enumerate(batch_files):

                img = cv2.imread(file)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                

                if transform is not None:

                    res = transform(image=img)

                    img = res['image']

                        #img = augment_and_mix(img)

                        

                if type!='train':

                    img = cv2.resize(img, shape[:2])

                    

                X_batch[i, :, :, 0] = img / 255.

            

            yield X_batch, [y_batch[:, i] for i in range(y_batch.shape[1])]

transforms_train = albumentations.Compose([

    RandomAugMix(severity=3, width=3, alpha=1., p=1.),

])



train_gen = data_generator(train_files, y_train,type='train',transform=transforms_train)

valid_gen = data_generator(valid_files, y_valid,type='test')



train_steps = round(len(train_files) / batch_size) + 1

valid_steps = round(len(valid_files) / batch_size) + 1
def build_model(mobilenet):

    x_in = layers.Input(shape=(128, 128, 1))

    x = layers.Conv2D(3, (3, 3), padding='same')(x_in)

    x = mobilenet(x)

    

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation='relu')(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.5)(x)

    

    out_grapheme = layers.Dense(168, activation='softmax', name='grapheme')(x)

    out_vowel = layers.Dense(11, activation='softmax', name='vowel')(x)

    out_consonant = layers.Dense(7, activation='softmax', name='consonant')(x)

    

    model = Model(inputs=x_in, outputs=[out_grapheme, out_vowel, out_consonant])

    

    model.compile(

        optimizers.Adam(lr=0.0001), 

        metrics=['accuracy'], 

        loss='sparse_categorical_crossentropy'

    )

    

    return model
mobilenet = MobileNetV2(include_top=False, weights=mobilenetpath+'/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5', input_shape=(128, 128, 3))

model = build_model(mobilenet)

model.summary()
callbacks = [tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)]



train_history = model.fit_generator(

    train_gen,

    steps_per_epoch=train_steps,

    epochs=5,

    validation_data=valid_gen,

    validation_steps=valid_steps,

    callbacks=callbacks

)
histories=pd.DataFrame(train_history.history)



plt.style.context("fivethirtyeight")



def plot_log(data, show=True):



    fig = plt.figure(figsize=(8,10))

    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)

    fig.add_subplot(211)

    

    for key in data.keys():

        if key.find('loss') >= 0:  # training loss

            plt.plot(data[key].values, label=key)

    plt.legend()

    plt.title('Training and Validtion Loss')



    fig.add_subplot(212)

    for key in data.keys():

        if key.find('acc') >= 0:  # acc

            plt.plot(data[key].values, label=key)

    plt.legend()

    plt.title('Training and Validation Accuracy')



    if show:

        plt.show()

        

plot_log(histories)