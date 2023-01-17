'''

import os

os.chdir('/kaggle/input/train-128')

!pwd

'''
'''

!unzip train_128_0.zip

!unzip train_128_1.zip

!unzip train_128_2.zip

!unzip train_128_3.zip

'''
import os



os.chdir('/kaggle/input/iterative-stratification')

!pip install './iterative_stratification-0.1.6-py3-none-any.whl'



os.chdir('/kaggle/input/kagglebengaliaihandwrittengraphemeclassification/KaggleKernelEfficientNetB3')

!pip install './efficientnet-1.0.0-py3-none-any.whl'



!pwd
import cv2

import os

import time, gc

import numpy as np

import pandas as pd



import tensorflow as tf

import keras

from keras import backend as K

from keras.models import Model, Input

from keras.layers import Dense, Lambda

from math import ceil



import efficientnet.keras as efn
import math

import random

import warnings

from PIL import Image

from glob import glob

import matplotlib.pyplot as plt

import seaborn as sns

import tqdm

import efficientnet.keras as efn

#sklearns 

from sklearn.metrics import cohen_kappa_score, accuracy_score

from sklearn.model_selection import train_test_split 



# keras modules 

from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from keras.optimizers import Adam, Nadam, SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model, load_model, Sequential

from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Conv2D, GlobalMaxPooling2D, concatenate

from keras.layers import (MaxPooling2D, Input, Average, Activation, MaxPool2D,

                          Flatten, LeakyReLU, BatchNormalization)

from keras import models

from keras import layers

from keras.preprocessing.image import load_img

from keras.preprocessing.image import img_to_array



from keras.utils import Sequence

from keras import utils as np_utils

from keras.callbacks import (Callback, ModelCheckpoint,

                                        LearningRateScheduler,EarlyStopping, 

                                        ReduceLROnPlateau,CSVLogger)



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.auto import tqdm

from glob import glob

from keras.utils import Sequence



from tensorflow import keras

import matplotlib.image as mpimg

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.models import clone_model

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization, Input

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import PIL.Image as Image, PIL.ImageDraw as ImageDraw, PIL.ImageFont as ImageFont

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Import Modules

import time, gc

from math import floor

# Keras

import keras.backend as K

from keras.optimizers import Adam

from keras.callbacks import Callback, ModelCheckpoint



# Iterative-Stratification

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

os.chdir('/kaggle/input/kagglebengaliaihandwrittengraphemeclassification/KaggleKernelEfficientNetB3')

# Custom 

from preprocessing import generate_images, resize_image

from model import create_model

from utils import plot_summaries
# Constants

HEIGHT = 137

WIDTH = 236

FACTOR = 0.70

HEIGHT_NEW = 128 #int(HEIGHT * FACTOR)

WIDTH_NEW = 128 #int(WIDTH * FACTOR)

CHANNELS = 3

BATCH_SIZE = 16



# Dir

DIR = '/kaggle/input/bengaliai-cv19' #'../input/bengaliai-cv19'
train_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/train.csv')

test_df_ = pd.read_csv('/kaggle/input/bengaliai-cv19/test.csv')

class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')

sample_sub_df = pd.read_csv('/kaggle/input/bengaliai-cv19/sample_submission.csv')
SEED = 2020

batch_size = 12 

# dim, size는 64x64 라서 64로 수정

dim = (128, 128)

SIZE = 128

stats = (0.0692, 0.2051)

HEIGHT = 137 

WIDTH = 236



import random

def seed_all(SEED):

    random.seed(SEED)

    np.random.seed(SEED)

    os.environ['PYTHONHASHSEED'] = str(SEED)

    

# seed all

seed_all(SEED)
import os

import cv2

import numpy as np

import pandas as pd

import albumentations

from albumentations.core.transforms_interface import DualTransform

from albumentations.augmentations import functional as F

import matplotlib.pyplot as plt

import torch

from torch.utils.data import TensorDataset, DataLoader, Dataset



## Grid Mask

# code takesn from https://www.kaggle.com/haqishen/gridmask



import albumentations

from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

from albumentations.augmentations import functional as F



class GridMask(DualTransform):

    """GridMask augmentation for image classification and object detection.



    Args:

        num_grid (int): number of grid in a row or column.

        fill_value (int, float, lisf of int, list of float): value for dropped pixels.

        rotate ((int, int) or int): range from which a random angle is picked. If rotate is a single int

            an angle is picked from (-rotate, rotate). Default: (-90, 90)

        mode (int):

            0 - cropout a quarter of the square of each grid (left top)

            1 - reserve a quarter of the square of each grid (left top)

            2 - cropout 2 quarter of the square of each grid (left top & right bottom)



    Targets:

        image, mask



    Image types:

        uint8, float32



    Reference:

    |  https://arxiv.org/abs/2001.04086

    |  https://github.com/akuxcw/GridMask

    """



    def __init__(self, num_grid=3, fill_value=0, rotate=0, mode=0, always_apply=False, p=0.5):

        super(GridMask, self).__init__(always_apply, p)

        if isinstance(num_grid, int):

            num_grid = (num_grid, num_grid)

        if isinstance(rotate, int):

            rotate = (-rotate, rotate)

        self.num_grid = num_grid

        self.fill_value = fill_value

        self.rotate = rotate

        self.mode = mode

        self.masks = None

        self.rand_h_max = []

        self.rand_w_max = []



    def init_masks(self, height, width):

        if self.masks is None:

            self.masks = []

            n_masks = self.num_grid[1] - self.num_grid[0] + 1

            for n, n_g in enumerate(range(self.num_grid[0], self.num_grid[1] + 1, 1)):

                grid_h = height / n_g

                grid_w = width / n_g

                this_mask = np.ones((int((n_g + 1) * grid_h), int((n_g + 1) * grid_w))).astype(np.uint8)

                for i in range(n_g + 1):

                    for j in range(n_g + 1):

                        this_mask[

                             int(i * grid_h) : int(i * grid_h + grid_h / 2),

                             int(j * grid_w) : int(j * grid_w + grid_w / 2)

                        ] = self.fill_value

                        if self.mode == 2:

                            this_mask[

                                 int(i * grid_h + grid_h / 2) : int(i * grid_h + grid_h),

                                 int(j * grid_w + grid_w / 2) : int(j * grid_w + grid_w)

                            ] = self.fill_value

                

                if self.mode == 1:

                    this_mask = 1 - this_mask



                self.masks.append(this_mask)

                self.rand_h_max.append(grid_h)

                self.rand_w_max.append(grid_w)



    def apply(self, image, mask, rand_h, rand_w, angle, **params):

        h, w = image.shape[:2]

        mask = F.rotate(mask, angle) if self.rotate[1] > 0 else mask

        mask = mask[:,:,np.newaxis] if image.ndim == 3 else mask

        image *= mask[rand_h:rand_h+h, rand_w:rand_w+w].astype(image.dtype)

        return image



    def get_params_dependent_on_targets(self, params):

        img = params['image']

        height, width = img.shape[:2]

        self.init_masks(height, width)



        mid = np.random.randint(len(self.masks))

        mask = self.masks[mid]

        rand_h = np.random.randint(self.rand_h_max[mid])

        rand_w = np.random.randint(self.rand_w_max[mid])

        angle = np.random.randint(self.rotate[0], self.rotate[1]) if self.rotate[1] > 0 else 0



        return {'mask': mask, 'rand_h': rand_h, 'rand_w': rand_w, 'angle': angle}



    @property

    def targets_as_params(self):

        return ['image']



    def get_transform_init_args_names(self):

        return ('num_grid', 'fill_value', 'rotate', 'mode')
# augmix : https://github.com/google-research/augmix



from PIL import Image

from PIL import ImageOps

import numpy as np



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

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, level, 0, 0, 1, 0),

                           resample=Image.BILINEAR)



def shear_y(pil_img, level):

    level = float_parameter(sample_level(level), 0.3)

    if np.random.uniform() > 0.5:

        level = -level

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, 0, 0, level, 1, 0),

                           resample=Image.BILINEAR)



def translate_x(pil_img, level):

    level = int_parameter(sample_level(level), SIZE / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, 0, level, 0, 1, 0),

                           resample=Image.BILINEAR)





def translate_y(pil_img, level):

    level = int_parameter(sample_level(level), SIZE / 3)

    if np.random.random() > 0.5:

        level = -level

    return pil_img.transform((SIZE, SIZE),

                           Image.AFFINE, (1, 0, 0, 0, 1, level),

                           resample=Image.BILINEAR)



augmentations = [

    autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y,

    translate_x, translate_y

]



# taken from https://www.kaggle.com/iafoss/image-preprocessing-128x128

MEAN = [ 0.06922848809290576,  0.06922848809290576,  0.06922848809290576]

STD = [ 0.20515700083327537,  0.20515700083327537,  0.20515700083327537]



def normalize(image):

    """Normalize input image channel-wise to zero mean and unit variance."""

    image = image.transpose(2, 0, 1)  # Switch to channel-first

    mean, std = np.array(MEAN), np.array(STD)

    image = (image - mean[:, None, None]) / std[:, None, None]

    return image.transpose(1, 2, 0)





def apply_op(image, op, severity):

    image = np.clip(image * 255., 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(image)  # Convert to PIL.Image

    pil_img = op(pil_img, severity)

    return np.asarray(pil_img) / 255.





def augment_and_mix(image, severity=1, width=3, depth=1, alpha=1.):

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



    mix = np.zeros_like(image)

    for i in range(width):

        image_aug = image.copy()

        depth = depth if depth > 0 else np.random.randint(1, 4)

        

        for _ in range(depth):

            op = np.random.choice(augmentations)

            image_aug = apply_op(image_aug, op, severity)

        mix = np.add(mix, ws[i] * normalize(image_aug), out=mix, 

                     casting="unsafe")



    mixed = (1 - m) * normalize(image) + m * mix

    return mixed
'''

# Image Size Summary

print(HEIGHT_NEW)

print(WIDTH_NEW)



# Image Prep

def resize_image(img, WIDTH_NEW, HEIGHT_NEW):

    # Invert

    img = 255 - img



    # Normalize

    img = (img * (255.0 / img.max())).astype(np.uint8)



    # Reshape

    img = img.reshape(HEIGHT, WIDTH)

    image_resized = cv2.resize(img, (WIDTH_NEW, HEIGHT_NEW), interpolation = cv2.INTER_AREA)



    return image_resized.reshape(-1)

'''   
# Generalized mean pool - GeM

gm_exp = tf.Variable(3.0, dtype = tf.float32)

def generalized_mean_pool_2d(X):

    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)),

                        axis = [1, 2], 

                        keepdims = False) + 1.e-7)**(1./gm_exp)

    return pool
from keras.models import model_from_json



# Create Model

def create_model(input_shape):

    '''

    # Input Layer

    input = Input(shape = input_shape)

    

    # Create and Compile Model and show Summary

    x_model = efn.EfficientNetB3(weights = None, include_top = False, input_tensor = input, pooling = None, classes = None)

    

    # UnFreeze all layers

    for layer in x_model.layers:

        layer.trainable = True

    

    # GeM

    lambda_layer = Lambda(generalized_mean_pool_2d)

    lambda_layer.trainable_weights.extend([gm_exp])

    x = lambda_layer(x_model.output)

    

    # multi output

    grapheme_root = Dense(168, activation = 'softmax', name = 'root')(x)

    vowel_diacritic = Dense(11, activation = 'softmax', name = 'vowel')(x)

    consonant_diacritic = Dense(7, activation = 'softmax', name = 'consonant')(x)



    # model

    model = Model(inputs = x_model.input, outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])

    '''

    os.chdir('/kaggle/input/json-string')

    model = model_from_json(json_string)

    return model
os.chdir('/kaggle/working')



# Import Modules

import os

import time, gc

import numpy as np

import pandas as pd

from math import floor

import cv2

import tensorflow as tf



# Keras

import keras

import keras.backend as K

from keras.optimizers import Adam

from keras.callbacks import Callback, ModelCheckpoint



# Iterative-Stratification

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit



# Custom 

from preprocessing import generate_images, resize_image

from model import create_model

from utils import plot_summaries



# Seeds

SEED = 1234

np.random.seed(SEED)

tf.random.set_seed(SEED)



# Input Dir

DATA_DIR = '/kaggle/input/bengaliai-cv19'     #'C:/KaggleBengaliAI/bengaliai-cv19'

TRAIN_DIR = '/kaggle/input/train-128-128/train_128_128/'         #'./train/'



# Constants

HEIGHT = 137

WIDTH = 236

SCALE_FACTOR = 0.70

HEIGHT_NEW = 128 #int(HEIGHT * SCALE_FACTOR)

WIDTH_NEW = 128 #int(WIDTH * SCALE_FACTOR)

RUN_NAME = 'Train2_'

PLOT_NAME1 = 'Train1_LossAndAccuracy.png'

PLOT_NAME2 = 'Train1_Recall.png'



BATCH_SIZE = 56

CHANNELS = 3

EPOCHS = 2

TEST_SIZE = 1./6



# Image Size Summary

#print(HEIGHT_NEW)

#print(WIDTH_NEW)



# Generate Image (Has to be done only one time .. or again when changing SCALE_FACTOR)

#GENERATE_IMAGES = True

#if GENERATE_IMAGES:

#    generate_images(DATA_DIR, TRAIN_DIR, WIDTH, HEIGHT, WIDTH_NEW, HEIGHT_NEW)



# Prepare Train Labels (Y)

train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

tgt_cols = ['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']

desc_df = train_df[tgt_cols].astype('str').describe()

types = desc_df.loc['unique',:]

X_train = train_df['image_id'].values

train_df = train_df[tgt_cols].astype('uint8')

for col in tgt_cols:

    train_df[col] = train_df[col].map('{:03}'.format)

Y_train = pd.get_dummies(train_df)



# Cleanup

del train_df

gc.collect()



# Modelcheckpoint

def ModelCheckpointFull(model_name):

    return ModelCheckpoint(model_name, 

                            monitor = 'val_loss', 

                            verbose = 1, 

                            save_best_only = False, 

                            save_weights_only = True, 

                            mode = 'min', 

                            period = 1)





def _read(path):

    img = cv2.imread(path)    

    return img



class TrainDataGenerator(keras.utils.Sequence):

#    def __init__(self, X_set, Y_set, ids, batch_size = 16, img_size = (512, 512, 3), img_dir = TRAIN_DIR, *args, **kwargs):

    def __init__(self, X_set, Y_set, ids, batch_size = 16, img_size = (128, 128, 3), img_dir = TRAIN_DIR, transform=None):

        self.X = X_set

        self.ids = ids

        self.Y = Y_set

        self.batch_size = batch_size

        self.img_size = img_size

        self.img_dir = img_dir

        self.on_epoch_end()

        self.transform = transform



        # Split Data

        self.x_indexed = self.X[self.ids]

        self.y_indexed = self.Y.iloc[self.ids]



        # Prep Y per Label   

        self.y_root = self.y_indexed.iloc[:,0:types['grapheme_root']].values

        self.y_vowel = self.y_indexed.iloc[:,types['grapheme_root']:types['grapheme_root']+types['vowel_diacritic']].values

        self.y_consonant = self.y_indexed.iloc[:,types['grapheme_root']+types['vowel_diacritic']:].values

    

    def __len__(self):

        return int(floor(len(self.ids) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        X, Y_root, Y_vowel, Y_consonant = self.__data_generation(indices)

        return X, {'root': Y_root, 'vowel': Y_vowel, 'consonant': Y_consonant}



    def on_epoch_end(self):

        self.indices = np.arange(len(self.ids))

    

    def __data_generation(self, indices):

        X = np.empty((self.batch_size, *self.img_size))

        Y_root = np.empty((self.batch_size, 168), dtype = np.int16)

        Y_vowel = np.empty((self.batch_size, 11), dtype = np.int16)

        Y_consonant = np.empty((self.batch_size, 7), dtype = np.int16)



        # Get Images for Batch

        for i, index in enumerate(indices):

            ID = self.x_indexed[index]

            image = _read(self.img_dir+ID+".png")

            image = cv2.resize(image,(128,128))#self._dim) 

            

            if self.transform is not None:

                if np.random.rand() > 0.7:

                    # albumentation : grid mask

                    res = self.transform(image=image)

                    image = res['image']

                else:

                    # augmix augmentation

                    image = augment_and_mix(image)

            

            # scaling  이거 지우면 scaling 안한 파일이 들어감

            image = (image.astype(np.float32)/255.0 - stats[0])/stats[1]

            

            # gray scaling 128x128

            gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114]) 

            image = gray(image) 

            

            # expand the axises , 128x128x3 만들기

            

            image = image.reshape(128,128,1)

            image = np.concatenate((image,)*3, axis=-1)





            X[i,] = image

        

        # Get Labels for Batch

        Y_root = self.y_root[indices]

        Y_vowel = self.y_vowel[indices]

        Y_consonant = self.y_consonant[indices]    

       

        return X, Y_root, Y_vowel, Y_consonant 



# Create Model

#model = create_model(input_shape = (HEIGHT_NEW, WIDTH_NEW, CHANNELS))

import json

from keras.models import model_from_json

model_json = "{\"class_name\": \"Model\", \"config\": {\"name\": \"efficientnet-b3\", \"layers\": [{\"name\": \"input_5\", \"class_name\": \"InputLayer\", \"config\": {\"batch_input_shape\": [null, 128, 128, 3], \"dtype\": \"float32\", \"sparse\": false, \"name\": \"input_5\"}, \"inbound_nodes\": []}, {\"name\": \"stem_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"stem_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 40, \"kernel_size\": [3, 3], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"input_5\", 0, 0, {}]]]}, {\"name\": \"stem_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"stem_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"stem_conv\", 0, 0, {}]]]}, {\"name\": \"stem_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"stem_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"stem_bn\", 0, 0, {}]]]}, {\"name\": \"block1a_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block1a_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"stem_activation\", 0, 0, {}]]]}, {\"name\": \"block1a_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block1a_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block1a_dwconv\", 0, 0, {}]]]}, {\"name\": \"block1a_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block1a_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block1a_bn\", 0, 0, {}]]]}, {\"name\": \"block1a_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block1a_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block1a_activation\", 0, 0, {}]]]}, {\"name\": \"block1a_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block1a_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 40]}, \"inbound_nodes\": [[[\"block1a_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block1a_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block1a_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 10, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block1a_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block1a_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block1a_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 40, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block1a_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block1a_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block1a_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block1a_activation\", 0, 0, {}], [\"block1a_se_expand\", 0, 0, {}]]]}, {\"name\": \"block1a_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block1a_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block1a_se_excite\", 0, 0, {}]]]}, {\"name\": \"block1a_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block1a_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block1a_project_conv\", 0, 0, {}]]]}, {\"name\": \"block1b_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block1b_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block1a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block1b_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block1b_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block1b_dwconv\", 0, 0, {}]]]}, {\"name\": \"block1b_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block1b_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block1b_bn\", 0, 0, {}]]]}, {\"name\": \"block1b_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block1b_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block1b_activation\", 0, 0, {}]]]}, {\"name\": \"block1b_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block1b_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 24]}, \"inbound_nodes\": [[[\"block1b_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block1b_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block1b_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 6, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block1b_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block1b_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block1b_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block1b_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block1b_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block1b_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block1b_activation\", 0, 0, {}], [\"block1b_se_expand\", 0, 0, {}]]]}, {\"name\": \"block1b_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block1b_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block1b_se_excite\", 0, 0, {}]]]}, {\"name\": \"block1b_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block1b_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block1b_project_conv\", 0, 0, {}]]]}, {\"name\": \"block1b_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block1b_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.0125, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block1b_project_bn\", 0, 0, {}]]]}, {\"name\": \"block1b_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block1b_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block1b_drop\", 0, 0, {}], [\"block1a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block2a_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2a_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 144, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block1b_add\", 0, 0, {}]]]}, {\"name\": \"block2a_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2a_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2a_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block2a_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block2a_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block2a_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block2a_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block2a_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block2a_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block2a_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2a_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2a_dwconv\", 0, 0, {}]]]}, {\"name\": \"block2a_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block2a_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block2a_bn\", 0, 0, {}]]]}, {\"name\": \"block2a_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block2a_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block2a_activation\", 0, 0, {}]]]}, {\"name\": \"block2a_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block2a_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 144]}, \"inbound_nodes\": [[[\"block2a_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block2a_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2a_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 6, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2a_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block2a_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2a_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 144, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2a_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block2a_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block2a_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block2a_activation\", 0, 0, {}], [\"block2a_se_expand\", 0, 0, {}]]]}, {\"name\": \"block2a_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2a_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 32, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2a_se_excite\", 0, 0, {}]]]}, {\"name\": \"block2a_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2a_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2a_project_conv\", 0, 0, {}]]]}, {\"name\": \"block2b_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2b_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 192, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block2b_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2b_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2b_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block2b_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block2b_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block2b_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block2b_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block2b_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block2b_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block2b_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2b_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2b_dwconv\", 0, 0, {}]]]}, {\"name\": \"block2b_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block2b_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block2b_bn\", 0, 0, {}]]]}, {\"name\": \"block2b_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block2b_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block2b_activation\", 0, 0, {}]]]}, {\"name\": \"block2b_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block2b_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 192]}, \"inbound_nodes\": [[[\"block2b_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block2b_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2b_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 8, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2b_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block2b_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2b_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 192, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2b_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block2b_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block2b_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block2b_activation\", 0, 0, {}], [\"block2b_se_expand\", 0, 0, {}]]]}, {\"name\": \"block2b_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2b_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 32, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2b_se_excite\", 0, 0, {}]]]}, {\"name\": \"block2b_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2b_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2b_project_conv\", 0, 0, {}]]]}, {\"name\": \"block2b_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block2b_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.037500000000000006, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block2b_project_bn\", 0, 0, {}]]]}, {\"name\": \"block2b_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block2b_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block2b_drop\", 0, 0, {}], [\"block2a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block2c_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2c_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 192, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2b_add\", 0, 0, {}]]]}, {\"name\": \"block2c_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2c_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2c_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block2c_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block2c_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block2c_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block2c_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block2c_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block2c_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block2c_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2c_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2c_dwconv\", 0, 0, {}]]]}, {\"name\": \"block2c_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block2c_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block2c_bn\", 0, 0, {}]]]}, {\"name\": \"block2c_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block2c_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block2c_activation\", 0, 0, {}]]]}, {\"name\": \"block2c_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block2c_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 192]}, \"inbound_nodes\": [[[\"block2c_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block2c_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2c_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 8, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2c_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block2c_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2c_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 192, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2c_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block2c_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block2c_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block2c_activation\", 0, 0, {}], [\"block2c_se_expand\", 0, 0, {}]]]}, {\"name\": \"block2c_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block2c_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 32, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2c_se_excite\", 0, 0, {}]]]}, {\"name\": \"block2c_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block2c_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block2c_project_conv\", 0, 0, {}]]]}, {\"name\": \"block2c_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block2c_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.05, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block2c_project_bn\", 0, 0, {}]]]}, {\"name\": \"block2c_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block2c_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block2c_drop\", 0, 0, {}], [\"block2b_add\", 0, 0, {}]]]}, {\"name\": \"block3a_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3a_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 192, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block2c_add\", 0, 0, {}]]]}, {\"name\": \"block3a_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3a_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3a_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block3a_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block3a_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block3a_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block3a_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block3a_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block3a_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block3a_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3a_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3a_dwconv\", 0, 0, {}]]]}, {\"name\": \"block3a_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block3a_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block3a_bn\", 0, 0, {}]]]}, {\"name\": \"block3a_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block3a_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block3a_activation\", 0, 0, {}]]]}, {\"name\": \"block3a_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block3a_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 192]}, \"inbound_nodes\": [[[\"block3a_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block3a_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3a_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 8, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3a_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block3a_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3a_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 192, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3a_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block3a_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block3a_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block3a_activation\", 0, 0, {}], [\"block3a_se_expand\", 0, 0, {}]]]}, {\"name\": \"block3a_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3a_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 48, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3a_se_excite\", 0, 0, {}]]]}, {\"name\": \"block3a_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3a_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3a_project_conv\", 0, 0, {}]]]}, {\"name\": \"block3b_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3b_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 288, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block3b_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3b_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3b_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block3b_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block3b_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block3b_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block3b_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block3b_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block3b_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block3b_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3b_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3b_dwconv\", 0, 0, {}]]]}, {\"name\": \"block3b_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block3b_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block3b_bn\", 0, 0, {}]]]}, {\"name\": \"block3b_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block3b_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block3b_activation\", 0, 0, {}]]]}, {\"name\": \"block3b_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block3b_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 288]}, \"inbound_nodes\": [[[\"block3b_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block3b_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3b_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 12, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3b_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block3b_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3b_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 288, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3b_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block3b_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block3b_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block3b_activation\", 0, 0, {}], [\"block3b_se_expand\", 0, 0, {}]]]}, {\"name\": \"block3b_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3b_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 48, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3b_se_excite\", 0, 0, {}]]]}, {\"name\": \"block3b_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3b_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3b_project_conv\", 0, 0, {}]]]}, {\"name\": \"block3b_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block3b_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.07500000000000001, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block3b_project_bn\", 0, 0, {}]]]}, {\"name\": \"block3b_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block3b_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block3b_drop\", 0, 0, {}], [\"block3a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block3c_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3c_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 288, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3b_add\", 0, 0, {}]]]}, {\"name\": \"block3c_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3c_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3c_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block3c_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block3c_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block3c_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block3c_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block3c_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block3c_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block3c_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3c_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3c_dwconv\", 0, 0, {}]]]}, {\"name\": \"block3c_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block3c_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block3c_bn\", 0, 0, {}]]]}, {\"name\": \"block3c_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block3c_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block3c_activation\", 0, 0, {}]]]}, {\"name\": \"block3c_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block3c_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 288]}, \"inbound_nodes\": [[[\"block3c_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block3c_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3c_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 12, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3c_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block3c_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3c_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 288, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3c_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block3c_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block3c_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block3c_activation\", 0, 0, {}], [\"block3c_se_expand\", 0, 0, {}]]]}, {\"name\": \"block3c_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block3c_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 48, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3c_se_excite\", 0, 0, {}]]]}, {\"name\": \"block3c_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block3c_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block3c_project_conv\", 0, 0, {}]]]}, {\"name\": \"block3c_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block3c_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.08750000000000001, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block3c_project_bn\", 0, 0, {}]]]}, {\"name\": \"block3c_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block3c_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block3c_drop\", 0, 0, {}], [\"block3b_add\", 0, 0, {}]]]}, {\"name\": \"block4a_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4a_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 288, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block3c_add\", 0, 0, {}]]]}, {\"name\": \"block4a_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4a_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4a_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block4a_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4a_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4a_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block4a_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block4a_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block4a_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block4a_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4a_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4a_dwconv\", 0, 0, {}]]]}, {\"name\": \"block4a_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4a_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4a_bn\", 0, 0, {}]]]}, {\"name\": \"block4a_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block4a_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block4a_activation\", 0, 0, {}]]]}, {\"name\": \"block4a_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block4a_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 288]}, \"inbound_nodes\": [[[\"block4a_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block4a_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4a_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 12, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4a_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block4a_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4a_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 288, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4a_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block4a_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block4a_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4a_activation\", 0, 0, {}], [\"block4a_se_expand\", 0, 0, {}]]]}, {\"name\": \"block4a_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4a_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 96, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4a_se_excite\", 0, 0, {}]]]}, {\"name\": \"block4a_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4a_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4a_project_conv\", 0, 0, {}]]]}, {\"name\": \"block4b_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4b_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block4b_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4b_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4b_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block4b_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4b_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4b_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block4b_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block4b_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block4b_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block4b_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4b_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4b_dwconv\", 0, 0, {}]]]}, {\"name\": \"block4b_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4b_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4b_bn\", 0, 0, {}]]]}, {\"name\": \"block4b_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block4b_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block4b_activation\", 0, 0, {}]]]}, {\"name\": \"block4b_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block4b_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 576]}, \"inbound_nodes\": [[[\"block4b_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block4b_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4b_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4b_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block4b_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4b_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4b_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block4b_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block4b_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4b_activation\", 0, 0, {}], [\"block4b_se_expand\", 0, 0, {}]]]}, {\"name\": \"block4b_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4b_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 96, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4b_se_excite\", 0, 0, {}]]]}, {\"name\": \"block4b_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4b_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4b_project_conv\", 0, 0, {}]]]}, {\"name\": \"block4b_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block4b_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.1125, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block4b_project_bn\", 0, 0, {}]]]}, {\"name\": \"block4b_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block4b_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4b_drop\", 0, 0, {}], [\"block4a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block4c_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4c_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4b_add\", 0, 0, {}]]]}, {\"name\": \"block4c_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4c_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4c_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block4c_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4c_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4c_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block4c_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block4c_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block4c_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block4c_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4c_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4c_dwconv\", 0, 0, {}]]]}, {\"name\": \"block4c_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4c_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4c_bn\", 0, 0, {}]]]}, {\"name\": \"block4c_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block4c_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block4c_activation\", 0, 0, {}]]]}, {\"name\": \"block4c_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block4c_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 576]}, \"inbound_nodes\": [[[\"block4c_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block4c_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4c_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4c_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block4c_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4c_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4c_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block4c_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block4c_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4c_activation\", 0, 0, {}], [\"block4c_se_expand\", 0, 0, {}]]]}, {\"name\": \"block4c_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4c_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 96, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4c_se_excite\", 0, 0, {}]]]}, {\"name\": \"block4c_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4c_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4c_project_conv\", 0, 0, {}]]]}, {\"name\": \"block4c_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block4c_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.125, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block4c_project_bn\", 0, 0, {}]]]}, {\"name\": \"block4c_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block4c_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4c_drop\", 0, 0, {}], [\"block4b_add\", 0, 0, {}]]]}, {\"name\": \"block4d_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4d_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4c_add\", 0, 0, {}]]]}, {\"name\": \"block4d_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4d_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4d_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block4d_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4d_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4d_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block4d_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block4d_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block4d_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block4d_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4d_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4d_dwconv\", 0, 0, {}]]]}, {\"name\": \"block4d_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4d_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4d_bn\", 0, 0, {}]]]}, {\"name\": \"block4d_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block4d_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block4d_activation\", 0, 0, {}]]]}, {\"name\": \"block4d_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block4d_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 576]}, \"inbound_nodes\": [[[\"block4d_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block4d_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4d_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4d_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block4d_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4d_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4d_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block4d_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block4d_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4d_activation\", 0, 0, {}], [\"block4d_se_expand\", 0, 0, {}]]]}, {\"name\": \"block4d_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4d_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 96, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4d_se_excite\", 0, 0, {}]]]}, {\"name\": \"block4d_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4d_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4d_project_conv\", 0, 0, {}]]]}, {\"name\": \"block4d_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block4d_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.1375, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block4d_project_bn\", 0, 0, {}]]]}, {\"name\": \"block4d_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block4d_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4d_drop\", 0, 0, {}], [\"block4c_add\", 0, 0, {}]]]}, {\"name\": \"block4e_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4e_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4d_add\", 0, 0, {}]]]}, {\"name\": \"block4e_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4e_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4e_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block4e_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4e_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4e_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block4e_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block4e_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block4e_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block4e_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4e_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4e_dwconv\", 0, 0, {}]]]}, {\"name\": \"block4e_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block4e_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block4e_bn\", 0, 0, {}]]]}, {\"name\": \"block4e_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block4e_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block4e_activation\", 0, 0, {}]]]}, {\"name\": \"block4e_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block4e_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 576]}, \"inbound_nodes\": [[[\"block4e_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block4e_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4e_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4e_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block4e_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4e_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4e_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block4e_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block4e_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4e_activation\", 0, 0, {}], [\"block4e_se_expand\", 0, 0, {}]]]}, {\"name\": \"block4e_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block4e_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 96, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4e_se_excite\", 0, 0, {}]]]}, {\"name\": \"block4e_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block4e_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block4e_project_conv\", 0, 0, {}]]]}, {\"name\": \"block4e_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block4e_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.15000000000000002, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block4e_project_bn\", 0, 0, {}]]]}, {\"name\": \"block4e_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block4e_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block4e_drop\", 0, 0, {}], [\"block4d_add\", 0, 0, {}]]]}, {\"name\": \"block5a_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5a_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block4e_add\", 0, 0, {}]]]}, {\"name\": \"block5a_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5a_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5a_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block5a_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5a_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5a_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block5a_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block5a_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block5a_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block5a_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5a_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5a_dwconv\", 0, 0, {}]]]}, {\"name\": \"block5a_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5a_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5a_bn\", 0, 0, {}]]]}, {\"name\": \"block5a_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block5a_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block5a_activation\", 0, 0, {}]]]}, {\"name\": \"block5a_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block5a_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 576]}, \"inbound_nodes\": [[[\"block5a_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block5a_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5a_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 24, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5a_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block5a_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5a_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 576, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5a_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block5a_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block5a_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5a_activation\", 0, 0, {}], [\"block5a_se_expand\", 0, 0, {}]]]}, {\"name\": \"block5a_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5a_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 136, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5a_se_excite\", 0, 0, {}]]]}, {\"name\": \"block5a_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5a_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5a_project_conv\", 0, 0, {}]]]}, {\"name\": \"block5b_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5b_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block5b_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5b_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5b_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block5b_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5b_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5b_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block5b_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block5b_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block5b_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block5b_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5b_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5b_dwconv\", 0, 0, {}]]]}, {\"name\": \"block5b_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5b_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5b_bn\", 0, 0, {}]]]}, {\"name\": \"block5b_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block5b_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block5b_activation\", 0, 0, {}]]]}, {\"name\": \"block5b_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block5b_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 816]}, \"inbound_nodes\": [[[\"block5b_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block5b_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5b_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 34, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5b_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block5b_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5b_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5b_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block5b_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block5b_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5b_activation\", 0, 0, {}], [\"block5b_se_expand\", 0, 0, {}]]]}, {\"name\": \"block5b_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5b_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 136, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5b_se_excite\", 0, 0, {}]]]}, {\"name\": \"block5b_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5b_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5b_project_conv\", 0, 0, {}]]]}, {\"name\": \"block5b_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block5b_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.17500000000000002, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block5b_project_bn\", 0, 0, {}]]]}, {\"name\": \"block5b_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block5b_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5b_drop\", 0, 0, {}], [\"block5a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block5c_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5c_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5b_add\", 0, 0, {}]]]}, {\"name\": \"block5c_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5c_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5c_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block5c_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5c_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5c_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block5c_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block5c_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block5c_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block5c_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5c_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5c_dwconv\", 0, 0, {}]]]}, {\"name\": \"block5c_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5c_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5c_bn\", 0, 0, {}]]]}, {\"name\": \"block5c_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block5c_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block5c_activation\", 0, 0, {}]]]}, {\"name\": \"block5c_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block5c_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 816]}, \"inbound_nodes\": [[[\"block5c_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block5c_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5c_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 34, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5c_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block5c_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5c_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5c_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block5c_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block5c_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5c_activation\", 0, 0, {}], [\"block5c_se_expand\", 0, 0, {}]]]}, {\"name\": \"block5c_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5c_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 136, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5c_se_excite\", 0, 0, {}]]]}, {\"name\": \"block5c_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5c_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5c_project_conv\", 0, 0, {}]]]}, {\"name\": \"block5c_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block5c_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.1875, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block5c_project_bn\", 0, 0, {}]]]}, {\"name\": \"block5c_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block5c_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5c_drop\", 0, 0, {}], [\"block5b_add\", 0, 0, {}]]]}, {\"name\": \"block5d_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5d_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5c_add\", 0, 0, {}]]]}, {\"name\": \"block5d_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5d_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5d_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block5d_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5d_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5d_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block5d_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block5d_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block5d_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block5d_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5d_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5d_dwconv\", 0, 0, {}]]]}, {\"name\": \"block5d_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5d_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5d_bn\", 0, 0, {}]]]}, {\"name\": \"block5d_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block5d_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block5d_activation\", 0, 0, {}]]]}, {\"name\": \"block5d_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block5d_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 816]}, \"inbound_nodes\": [[[\"block5d_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block5d_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5d_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 34, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5d_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block5d_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5d_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5d_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block5d_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block5d_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5d_activation\", 0, 0, {}], [\"block5d_se_expand\", 0, 0, {}]]]}, {\"name\": \"block5d_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5d_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 136, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5d_se_excite\", 0, 0, {}]]]}, {\"name\": \"block5d_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5d_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5d_project_conv\", 0, 0, {}]]]}, {\"name\": \"block5d_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block5d_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.2, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block5d_project_bn\", 0, 0, {}]]]}, {\"name\": \"block5d_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block5d_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5d_drop\", 0, 0, {}], [\"block5c_add\", 0, 0, {}]]]}, {\"name\": \"block5e_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5e_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5d_add\", 0, 0, {}]]]}, {\"name\": \"block5e_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5e_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5e_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block5e_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5e_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5e_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block5e_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block5e_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block5e_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block5e_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5e_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5e_dwconv\", 0, 0, {}]]]}, {\"name\": \"block5e_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block5e_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block5e_bn\", 0, 0, {}]]]}, {\"name\": \"block5e_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block5e_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block5e_activation\", 0, 0, {}]]]}, {\"name\": \"block5e_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block5e_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 816]}, \"inbound_nodes\": [[[\"block5e_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block5e_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5e_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 34, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5e_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block5e_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5e_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5e_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block5e_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block5e_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5e_activation\", 0, 0, {}], [\"block5e_se_expand\", 0, 0, {}]]]}, {\"name\": \"block5e_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block5e_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 136, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5e_se_excite\", 0, 0, {}]]]}, {\"name\": \"block5e_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block5e_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block5e_project_conv\", 0, 0, {}]]]}, {\"name\": \"block5e_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block5e_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.21250000000000002, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block5e_project_bn\", 0, 0, {}]]]}, {\"name\": \"block5e_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block5e_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block5e_drop\", 0, 0, {}], [\"block5d_add\", 0, 0, {}]]]}, {\"name\": \"block6a_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6a_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block5e_add\", 0, 0, {}]]]}, {\"name\": \"block6a_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6a_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6a_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block6a_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6a_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6a_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block6a_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block6a_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [2, 2], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block6a_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block6a_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6a_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6a_dwconv\", 0, 0, {}]]]}, {\"name\": \"block6a_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6a_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6a_bn\", 0, 0, {}]]]}, {\"name\": \"block6a_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block6a_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block6a_activation\", 0, 0, {}]]]}, {\"name\": \"block6a_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block6a_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 816]}, \"inbound_nodes\": [[[\"block6a_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block6a_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6a_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 34, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6a_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block6a_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6a_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 816, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6a_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block6a_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block6a_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6a_activation\", 0, 0, {}], [\"block6a_se_expand\", 0, 0, {}]]]}, {\"name\": \"block6a_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6a_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 232, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6a_se_excite\", 0, 0, {}]]]}, {\"name\": \"block6a_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6a_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6a_project_conv\", 0, 0, {}]]]}, {\"name\": \"block6b_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6b_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block6b_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6b_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6b_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block6b_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6b_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6b_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block6b_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block6b_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block6b_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block6b_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6b_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6b_dwconv\", 0, 0, {}]]]}, {\"name\": \"block6b_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6b_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6b_bn\", 0, 0, {}]]]}, {\"name\": \"block6b_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block6b_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block6b_activation\", 0, 0, {}]]]}, {\"name\": \"block6b_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block6b_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 1392]}, \"inbound_nodes\": [[[\"block6b_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block6b_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6b_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 58, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6b_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block6b_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6b_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6b_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block6b_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block6b_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6b_activation\", 0, 0, {}], [\"block6b_se_expand\", 0, 0, {}]]]}, {\"name\": \"block6b_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6b_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 232, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6b_se_excite\", 0, 0, {}]]]}, {\"name\": \"block6b_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6b_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6b_project_conv\", 0, 0, {}]]]}, {\"name\": \"block6b_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block6b_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.23750000000000002, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block6b_project_bn\", 0, 0, {}]]]}, {\"name\": \"block6b_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block6b_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6b_drop\", 0, 0, {}], [\"block6a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block6c_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6c_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6b_add\", 0, 0, {}]]]}, {\"name\": \"block6c_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6c_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6c_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block6c_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6c_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6c_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block6c_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block6c_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block6c_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block6c_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6c_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6c_dwconv\", 0, 0, {}]]]}, {\"name\": \"block6c_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6c_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6c_bn\", 0, 0, {}]]]}, {\"name\": \"block6c_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block6c_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block6c_activation\", 0, 0, {}]]]}, {\"name\": \"block6c_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block6c_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 1392]}, \"inbound_nodes\": [[[\"block6c_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block6c_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6c_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 58, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6c_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block6c_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6c_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6c_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block6c_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block6c_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6c_activation\", 0, 0, {}], [\"block6c_se_expand\", 0, 0, {}]]]}, {\"name\": \"block6c_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6c_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 232, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6c_se_excite\", 0, 0, {}]]]}, {\"name\": \"block6c_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6c_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6c_project_conv\", 0, 0, {}]]]}, {\"name\": \"block6c_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block6c_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.25, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block6c_project_bn\", 0, 0, {}]]]}, {\"name\": \"block6c_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block6c_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6c_drop\", 0, 0, {}], [\"block6b_add\", 0, 0, {}]]]}, {\"name\": \"block6d_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6d_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6c_add\", 0, 0, {}]]]}, {\"name\": \"block6d_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6d_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6d_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block6d_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6d_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6d_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block6d_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block6d_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block6d_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block6d_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6d_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6d_dwconv\", 0, 0, {}]]]}, {\"name\": \"block6d_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6d_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6d_bn\", 0, 0, {}]]]}, {\"name\": \"block6d_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block6d_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block6d_activation\", 0, 0, {}]]]}, {\"name\": \"block6d_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block6d_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 1392]}, \"inbound_nodes\": [[[\"block6d_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block6d_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6d_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 58, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6d_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block6d_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6d_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6d_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block6d_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block6d_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6d_activation\", 0, 0, {}], [\"block6d_se_expand\", 0, 0, {}]]]}, {\"name\": \"block6d_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6d_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 232, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6d_se_excite\", 0, 0, {}]]]}, {\"name\": \"block6d_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6d_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6d_project_conv\", 0, 0, {}]]]}, {\"name\": \"block6d_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block6d_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.2625, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block6d_project_bn\", 0, 0, {}]]]}, {\"name\": \"block6d_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block6d_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6d_drop\", 0, 0, {}], [\"block6c_add\", 0, 0, {}]]]}, {\"name\": \"block6e_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6e_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6d_add\", 0, 0, {}]]]}, {\"name\": \"block6e_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6e_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6e_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block6e_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6e_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6e_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block6e_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block6e_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block6e_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block6e_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6e_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6e_dwconv\", 0, 0, {}]]]}, {\"name\": \"block6e_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6e_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6e_bn\", 0, 0, {}]]]}, {\"name\": \"block6e_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block6e_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block6e_activation\", 0, 0, {}]]]}, {\"name\": \"block6e_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block6e_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 1392]}, \"inbound_nodes\": [[[\"block6e_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block6e_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6e_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 58, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6e_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block6e_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6e_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6e_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block6e_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block6e_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6e_activation\", 0, 0, {}], [\"block6e_se_expand\", 0, 0, {}]]]}, {\"name\": \"block6e_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6e_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 232, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6e_se_excite\", 0, 0, {}]]]}, {\"name\": \"block6e_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6e_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6e_project_conv\", 0, 0, {}]]]}, {\"name\": \"block6e_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block6e_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.275, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block6e_project_bn\", 0, 0, {}]]]}, {\"name\": \"block6e_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block6e_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6e_drop\", 0, 0, {}], [\"block6d_add\", 0, 0, {}]]]}, {\"name\": \"block6f_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6f_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6e_add\", 0, 0, {}]]]}, {\"name\": \"block6f_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6f_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6f_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block6f_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6f_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6f_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block6f_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block6f_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [5, 5], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block6f_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block6f_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6f_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6f_dwconv\", 0, 0, {}]]]}, {\"name\": \"block6f_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block6f_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block6f_bn\", 0, 0, {}]]]}, {\"name\": \"block6f_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block6f_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block6f_activation\", 0, 0, {}]]]}, {\"name\": \"block6f_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block6f_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 1392]}, \"inbound_nodes\": [[[\"block6f_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block6f_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6f_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 58, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6f_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block6f_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6f_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6f_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block6f_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block6f_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6f_activation\", 0, 0, {}], [\"block6f_se_expand\", 0, 0, {}]]]}, {\"name\": \"block6f_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block6f_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 232, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6f_se_excite\", 0, 0, {}]]]}, {\"name\": \"block6f_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block6f_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block6f_project_conv\", 0, 0, {}]]]}, {\"name\": \"block6f_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block6f_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.28750000000000003, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block6f_project_bn\", 0, 0, {}]]]}, {\"name\": \"block6f_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block6f_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block6f_drop\", 0, 0, {}], [\"block6e_add\", 0, 0, {}]]]}, {\"name\": \"block7a_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7a_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block6f_add\", 0, 0, {}]]]}, {\"name\": \"block7a_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block7a_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block7a_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block7a_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block7a_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block7a_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block7a_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block7a_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block7a_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block7a_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block7a_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block7a_dwconv\", 0, 0, {}]]]}, {\"name\": \"block7a_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block7a_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block7a_bn\", 0, 0, {}]]]}, {\"name\": \"block7a_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block7a_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block7a_activation\", 0, 0, {}]]]}, {\"name\": \"block7a_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block7a_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 1392]}, \"inbound_nodes\": [[[\"block7a_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block7a_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7a_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 58, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7a_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block7a_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7a_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1392, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7a_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block7a_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block7a_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block7a_activation\", 0, 0, {}], [\"block7a_se_expand\", 0, 0, {}]]]}, {\"name\": \"block7a_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7a_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 384, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7a_se_excite\", 0, 0, {}]]]}, {\"name\": \"block7a_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block7a_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block7a_project_conv\", 0, 0, {}]]]}, {\"name\": \"block7b_expand_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7b_expand_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 2304, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7a_project_bn\", 0, 0, {}]]]}, {\"name\": \"block7b_expand_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block7b_expand_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block7b_expand_conv\", 0, 0, {}]]]}, {\"name\": \"block7b_expand_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block7b_expand_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block7b_expand_bn\", 0, 0, {}]]]}, {\"name\": \"block7b_dwconv\", \"class_name\": \"DepthwiseConv2D\", \"config\": {\"name\": \"block7b_dwconv\", \"trainable\": true, \"dtype\": \"float32\", \"kernel_size\": [3, 3], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"bias_regularizer\": null, \"activity_regularizer\": null, \"bias_constraint\": null, \"depth_multiplier\": 1, \"depthwise_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"depthwise_regularizer\": null, \"depthwise_constraint\": null}, \"inbound_nodes\": [[[\"block7b_expand_activation\", 0, 0, {}]]]}, {\"name\": \"block7b_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block7b_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block7b_dwconv\", 0, 0, {}]]]}, {\"name\": \"block7b_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"block7b_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"block7b_bn\", 0, 0, {}]]]}, {\"name\": \"block7b_se_squeeze\", \"class_name\": \"GlobalAveragePooling2D\", \"config\": {\"name\": \"block7b_se_squeeze\", \"trainable\": true, \"dtype\": \"float32\", \"data_format\": \"channels_last\"}, \"inbound_nodes\": [[[\"block7b_activation\", 0, 0, {}]]]}, {\"name\": \"block7b_se_reshape\", \"class_name\": \"Reshape\", \"config\": {\"name\": \"block7b_se_reshape\", \"trainable\": true, \"dtype\": \"float32\", \"target_shape\": [1, 1, 2304]}, \"inbound_nodes\": [[[\"block7b_se_squeeze\", 0, 0, {}]]]}, {\"name\": \"block7b_se_reduce\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7b_se_reduce\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 96, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"swish\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7b_se_reshape\", 0, 0, {}]]]}, {\"name\": \"block7b_se_expand\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7b_se_expand\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 2304, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"sigmoid\", \"use_bias\": true, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7b_se_reduce\", 0, 0, {}]]]}, {\"name\": \"block7b_se_excite\", \"class_name\": \"Multiply\", \"config\": {\"name\": \"block7b_se_excite\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block7b_activation\", 0, 0, {}], [\"block7b_se_expand\", 0, 0, {}]]]}, {\"name\": \"block7b_project_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"block7b_project_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 384, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7b_se_excite\", 0, 0, {}]]]}, {\"name\": \"block7b_project_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"block7b_project_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"block7b_project_conv\", 0, 0, {}]]]}, {\"name\": \"block7b_drop\", \"class_name\": \"FixedDropout\", \"config\": {\"name\": \"block7b_drop\", \"trainable\": true, \"dtype\": \"float32\", \"rate\": 0.3125, \"noise_shape\": [null, 1, 1, 1], \"seed\": null}, \"inbound_nodes\": [[[\"block7b_project_bn\", 0, 0, {}]]]}, {\"name\": \"block7b_add\", \"class_name\": \"Add\", \"config\": {\"name\": \"block7b_add\", \"trainable\": true, \"dtype\": \"float32\"}, \"inbound_nodes\": [[[\"block7b_drop\", 0, 0, {}], [\"block7a_project_bn\", 0, 0, {}]]]}, {\"name\": \"top_conv\", \"class_name\": \"Conv2D\", \"config\": {\"name\": \"top_conv\", \"trainable\": true, \"dtype\": \"float32\", \"filters\": 1536, \"kernel_size\": [1, 1], \"strides\": [1, 1], \"padding\": \"same\", \"data_format\": \"channels_last\", \"dilation_rate\": [1, 1], \"activation\": \"linear\", \"use_bias\": false, \"kernel_initializer\": {\"class_name\": \"VarianceScaling\", \"config\": {\"scale\": 2.0, \"mode\": \"fan_out\", \"distribution\": \"normal\", \"seed\": null}}, \"bias_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"kernel_regularizer\": null, \"bias_regularizer\": null, \"activity_regularizer\": null, \"kernel_constraint\": null, \"bias_constraint\": null}, \"inbound_nodes\": [[[\"block7b_add\", 0, 0, {}]]]}, {\"name\": \"top_bn\", \"class_name\": \"BatchNormalization\", \"config\": {\"name\": \"top_bn\", \"trainable\": true, \"dtype\": \"float32\", \"axis\": 3, \"momentum\": 0.99, \"epsilon\": 0.001, \"center\": true, \"scale\": true, \"beta_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"gamma_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"moving_mean_initializer\": {\"class_name\": \"Zeros\", \"config\": {}}, \"moving_variance_initializer\": {\"class_name\": \"Ones\", \"config\": {}}, \"beta_regularizer\": null, \"gamma_regularizer\": null, \"beta_constraint\": null, \"gamma_constraint\": null}, \"inbound_nodes\": [[[\"top_conv\", 0, 0, {}]]]}, {\"name\": \"top_activation\", \"class_name\": \"Activation\", \"config\": {\"name\": \"top_activation\", \"trainable\": true, \"dtype\": \"float32\", \"activation\": \"swish\"}, \"inbound_nodes\": [[[\"top_bn\", 0, 0, {}]]]}], \"input_layers\": [[\"input_5\", 0, 0]], \"output_layers\": [[\"top_activation\", 0, 0]]}, \"keras_version\": \"2.3.1\", \"backend\": \"tensorflow\"}"

json_string = json.loads(json.dumps(model_json))

x_model = model_from_json(json_string)



# Generalized mean pool - GeM

gm_exp = tf.Variable(3.0, dtype = tf.float32)

def generalized_mean_pool_2d(X):

    pool = (tf.reduce_mean(tf.abs(X**(gm_exp)),

                        axis = [1, 2], 

                        keepdims = False) + 1.e-7)**(1./gm_exp)

    return pool



# UnFreeze all layers

for layer in x_model.layers:

    layer.trainable = True

    

    # GeM

lambda_layer = Lambda(generalized_mean_pool_2d)

lambda_layer.trainable_weights.extend([gm_exp])

x = lambda_layer(x_model.output)

    

    # multi output

grapheme_root = Dense(168, activation = 'softmax', name = 'root')(x)

vowel_diacritic = Dense(11, activation = 'softmax', name = 'vowel')(x)

consonant_diacritic = Dense(7, activation = 'softmax', name = 'consonant')(x)



    # model

model = Model(inputs = x_model.input, outputs = [grapheme_root, vowel_diacritic, consonant_diacritic])



# Compile Model

model.compile(optimizer = Adam(lr = 0.00016),

                loss = {'root': 'categorical_crossentropy',

                        'vowel': 'categorical_crossentropy',

                        'consonant': 'categorical_crossentropy'},

                loss_weights = {'root': 0.50,        

                                'vowel': 0.25,

                                'consonant': 0.25},

                metrics = {'root': ['accuracy', tf.keras.metrics.Recall()],

                            'vowel': ['accuracy', tf.keras.metrics.Recall()],

                            'consonant': ['accuracy', tf.keras.metrics.Recall()] })



from keras.models import load_model

#model = load_model('/kaggle/input/kagglebengaliaihandwrittengraphemeclassification/KaggleKernelEfficientNetB3/model_weights/Train1_model_59.h5')

model.load_weights('/kaggle/input/bg-yhm68/Train2_model_68.h5')



# Model Summary

print(model.summary())



# Multi Label Stratified Split stuff...

msss = MultilabelStratifiedShuffleSplit(n_splits = EPOCHS, test_size = TEST_SIZE, random_state = SEED)



# CustomReduceLRonPlateau function

best_val_loss = np.Inf

def CustomReduceLRonPlateau(model, history, epoch):

    global best_val_loss

    

    # ReduceLR Constants

    monitor = 'val_root_loss'

    patience = 5

    factor = 0.75

    min_lr = 1e-5



    # Get Current LR

    current_lr = float(K.get_value(model.optimizer.lr))

    

    # Print Current Learning Rate

    print('Current LR: {0}'.format(current_lr))



    # Monitor Best Value

    current_val_loss = history[monitor][-1]

    if current_val_loss < best_val_loss:

        best_val_loss = current_val_loss

    print('Best Vall Loss: {0}'.format(best_val_loss))



    # Track last values

    if len(history[monitor]) >= patience:

        last5 = history[monitor][-5:]

        print('Last: {0}'.format(last5))

        best_in_last = min(last5)

        print('Min value in Last: {0}'.format(best_in_last))



        # Determine correction

        if best_val_loss < best_in_last:

            new_lr = current_lr * factor

            if new_lr < min_lr:

                new_lr = min_lr

            print('ReduceLRonPlateau setting learning rate to: {0}'.format(new_lr))

            K.set_value(model.optimizer.lr, new_lr)



# History Placeholder

history = {}



# Epoch Training Loop

for epoch, msss_splits in zip(range(0, EPOCHS), msss.split(X_train, Y_train)):

    print('=========== EPOCH {}'.format(epoch+69))



    # Get train and test index, shuffle train indexes.

    train_idx = msss_splits[0]

    valid_idx = msss_splits[1]

    np.random.shuffle(train_idx)

    print('Train Length: {0}   First 10 indices: {1}'.format(len(train_idx), train_idx[:10]))    

    print('Valid Length: {0}    First 10 indices: {1}'.format(len(valid_idx), valid_idx[:10]))



    transforms_train = albumentations.Compose([

        GridMask(num_grid=3, rotate=15, p=1),

    ])



    # Create Data Generators for Train and Valid

    data_generator_train = TrainDataGenerator(X_train, 

                                            Y_train,

                                            train_idx, 

                                            BATCH_SIZE, 

                                            (HEIGHT_NEW, WIDTH_NEW, CHANNELS),

                                            img_dir = TRAIN_DIR, transform=transforms_train)

    data_generator_val = TrainDataGenerator(X_train, 

                                            Y_train,

                                            valid_idx,

                                            BATCH_SIZE, 

                                            (HEIGHT_NEW, WIDTH_NEW, CHANNELS),

                                            img_dir = TRAIN_DIR)



    TRAIN_STEPS = int(len(data_generator_train))

    VALID_STEPS = int(len(data_generator_val))

    print('Train Generator Size: {0}'.format(len(data_generator_train)))

    print('Validation Generator Size: {0}'.format(len(data_generator_val)))

    

    model.fit_generator(generator = data_generator_train,

                        validation_data = data_generator_val,

                        steps_per_epoch = TRAIN_STEPS,

                        validation_steps = VALID_STEPS,

                        epochs = 1,

                        callbacks = [ModelCheckpointFull(RUN_NAME + 'model_' + str(epoch+69) + '.h5')],

                        verbose = 1)



    # Set and Concat Training History

    temp_history = model.history.history

    if epoch == 0:

        history = temp_history

    else:

        for k in temp_history: history[k] = history[k] + temp_history[k]



    # Custom ReduceLRonPlateau

    CustomReduceLRonPlateau(model, history, epoch)



    # Cleanup

    del data_generator_train, data_generator_val, train_idx, valid_idx

    gc.collect()



# Plot Training Summaries

plot_summaries(history, PLOT_NAME1, PLOT_NAME2)



# Create Predictions

row_ids, targets = [], []

id = 0



# Loop through parquet files

for i in range(4):

    img_df = pd.read_parquet(os.path.join(DATA_DIR, 'test_image_data_'+str(i)+'.parquet'))

    img_df = img_df.drop('image_id', axis = 1)

    

    # Loop through rows in parquet file

    for index, row in img_df.iterrows():

        img = resize_image(row.values, WIDTH, HEIGHT, WIDTH_NEW, HEIGHT_NEW)

        img = np.stack((img,)*CHANNELS, axis=-1)

        image = img.reshape(-1, HEIGHT_NEW, WIDTH_NEW, 3)

        

        # Predict

        preds = model.predict(image, verbose = 1)

        for k in range(3):

            row_ids.append('Test_' + str(id) + '_' + tgt_cols[k])

            targets.append(np.argmax(preds[k]))

        id += 1



# Create and Save Submission File

submission = pd.DataFrame({'row_id': row_ids, 'target': targets}, columns = ['row_id', 'target'])

submission.to_csv('submission.csv', index = False)

print(submission.head(25))