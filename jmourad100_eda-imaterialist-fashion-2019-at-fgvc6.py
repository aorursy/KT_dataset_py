##############

## Loading dependencies

############################

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os, time, random, cv2, tqdm

from imgaug import augmenters as iaa



import tensorflow as tf

from keras import backend as K

from keras.models import Model

from keras.applications import (Xception, VGG16, VGG19, ResNet50, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, NASNetMobile)

from keras.layers import Conv2D, Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, concatenate

from keras.layers import Activation, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import SGD, Adam, RMSprop

from keras.metrics import *

from keras.utils.np_utils import to_categorical

from keras.utils import Sequence

from keras import regularizers



from keras.utils.vis_utils import plot_model

from IPython.display import Image



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.utils import class_weight, shuffle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



! ls ../input/
###########

###### Configuration

##############################

%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import json



# Reading the json as a dict

with open('../input/label_descriptions.json') as json_data:

    data = json.load(json_data)
JSON_COLUMNS = list(data.keys())
JSON_COLUMNS
data
from pandas.io.json import json_normalize

df = json_normalize(data)
df.head()