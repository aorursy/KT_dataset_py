# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session




import json

import math

import os



import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score, auc, roc_auc_score, roc_curve

import sklearn

import scipy

import tensorflow as tf

from tqdm import tqdm

from keras.preprocessing import image

from keras.models import Model

from keras.layers import BatchNormalization, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense



%matplotlib inline



!pip install git+https://github.com/qubvel/efficientnet
from efficientnet.tfkeras import EfficientNetB7 as effnetb7
np.random.seed(2019)

tf.random.set_seed(2019)

TEST_SIZE = 0.40

SEED = 2019

BATCH_SIZE = 8
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

train_df.head(7)
x_train = np.load('../input/four-fold-aptos/train_all_four.npy')

x_test = np.load('../input/four-fold-aptos/test_all_four.npy')
y_train = train_df['diagnosis'].values

y_train

y_train_one_hot = pd.get_dummies(train_df['diagnosis']).values



y_train_multi = np.empty(y_train_one_hot.shape, dtype=y_train_one_hot.dtype)

y_train_multi[:, 4] = y_train_one_hot[:, 4]



for i in range(3, -1, -1):

    y_train_multi[:, i] = np.logical_or(y_train_one_hot[:, i], y_train_multi[:, i+1])



print("Original y_train:", y_train_one_hot.sum(axis=0))

print("Multilabel version:", y_train_multi.sum(axis=0))
x_train, x_val, y_train, y_val = train_test_split(

    x_train, y_train_multi, 

    test_size=TEST_SIZE, 

    random_state=SEED

)
def create_datagen():

    return ImageDataGenerator(

        zoom_range=0.15,  # set range for random zoom

        # set mode for filling points outside the input boundaries

        fill_mode='constant',

        cval=0.,  # value used for fill_mode = "constant"

        horizontal_flip=True,  # randomly flip images

        vertical_flip=True,  # randomly flip images

    )



# Using original generator

data_generator = create_datagen().flow(x_train, y_train, batch_size=BATCH_SIZE, seed=SEED)
def vgg19():

    base_model = tf.keras.applications.VGG19(include_top=False,

                                            weights="imagenet",

                                            input_shape=x_train[0].shape)

    x = base_model.output

    batch_normal = BatchNormalization()(x)

    global_avg_pooling = GlobalAveragePooling2D()(batch_normal)

    drop_out = Dropout(0.5)(global_avg_pooling)

    dense1 = Dense(1024, activation='relu')(drop_out)

    dense2 = Dense(5, activation = 'sigmoid')(dense1)

    model = Model(inputs = base_model.input, outputs = dense2)

    return model
def resnet50():

    base_model = tf.keras.applications.ResNet50(include_top=False,

                                            weights="imagenet",

                                            input_shape=x_train[0].shape)

    x = base_model.output

    batch_normal = BatchNormalization()(x)

    global_avg_pooling = GlobalAveragePooling2D()(batch_normal)

    drop_out = Dropout(0.5)(global_avg_pooling)

    dense1 = Dense(1024, activation='relu')(drop_out)

    dense2 = Dense(5, activation = 'sigmoid')(dense1)

    model = Model(inputs = base_model.input, outputs = dense2)

    return model
def efficientnetb7():

    base_model = effnetb7(include_top=False,

                     weights = None,

                     input_shape=(224,224,3))

    x = base_model.output

    batch_normal = BatchNormalization()(x)

    global_avg_pooling = GlobalAveragePooling2D()(batch_normal)

    drop_out = Dropout(0.5)(global_avg_pooling)

    dense1 = Dense(1024, activation='relu')(drop_out)

    dense2 = Dense(5, activation = 'sigmoid')(dense1)

    model = Model(inputs = base_model.input, outputs = dense2)

    return model
def densenet121():

    densenet = DenseNet121(weights=None, include_top=False, input_shape=(224,224,3))

    x = densenet.output

    batch_normal = BatchNormalization()(x)

    global_avg_pooling = GlobalAveragePooling2D()(batch_normal)

    drop_out = Dropout(0.5)(global_avg_pooling)

    dense1 = Dense(1024, activation='relu')(drop_out)

    dense2 = Dense(5, activation = 'sigmoid')(dense1)

    model = Model(inputs = densenet.input, outputs = dense2)

    return model
vgg = vgg19()

vgg.load_weights('../input/resnet50-weights-aptos/vgg19_aptos.h5')
resnet = resnet50()

resnet.load_weights('../input/all-nets-aptos/resnet50_aptos.h5')
effnet = efficientnetb7()

effnet.load_weights('../input/effnet-weights-aptos/efficientnet-b7_noisy_student_notop_four_fold_preprocess_aptos.h5')
densenet = densenet121()

densenet.load_weights('../input/aptos-densenet121/densenet121.h5')
eff_pred = effnet.predict(x_val)
def multi_label(x):

    val_y = x > 0.5

    val_y = val_y.astype(int).sum(axis=1) - 1

    return val_y
eff_label = multi_label(eff_pred)
res_pred = resnet.predict(x_val)
res_label = multi_label(res_pred)
vgg_pred = vgg.predict(x_val)
vgg_label = multi_label(vgg_pred)
dense_pred = densenet.predict(x_val)
dense_label = multi_label(dense_pred)
train_df.iloc[[20,21]]['diagnosis']
pd.get_dummies(vgg_label).values
vgg_label
res_label
y_real = [4 if (list(i)[4]==1) else list(i).index(0)-1 for i in y_val]
import math
f = (eff_label + vgg_label + res_label + dense_label)/4
final = []

for i in f:

    final.append(math.ceil(i))
final
y_real
cohen_kappa_score(

            y_real,

            final, 

            weights='quadratic'

        )
from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report



actual = y_real

predicted = final

results = confusion_matrix(actual, predicted) 

  

print ('Confusion Matrix :')

print(results)

print ('Accuracy Score :',accuracy_score(actual, predicted) )

print ('Report : ')

print (classification_report(actual, predicted))