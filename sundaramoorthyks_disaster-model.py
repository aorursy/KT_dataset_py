import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import  DenseNet201

from keras.layers.core import Dropout

from keras.layers.core import Flatten

from keras.layers.core import Dense

from keras.layers import Input

from keras.models import Model

from keras.optimizers import SGD

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import numpy as np

import pickle

import os

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import  DenseNet201

from keras.models import Sequential, Model, load_model

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from keras import regularizers

from keras import backend as K



from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from PIL import Image

import seaborn as sns



#callback need

from sklearn.datasets.samples_generator import make_blobs

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import Callback

from keras.optimizers import SGD

from keras import backend

from math import pi

from math import cos

from math import floor

%load_ext tensorboard



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
model = load_model('/kaggle/input/disasterwithoutearthquake/Densenet_without_earthquake.h5')

print(model.summary())
test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(

    directory="/kaggle/input/withoutearthquake/edit-new-images-without-earthquake/test",

    target_size=(224 ,224),

    color_mode="rgb",

    batch_size=32,

    class_mode=None,

    shuffle=False

)

test_generator.class_indices
test_generator.reset()

pred=model.predict_generator(test_generator,verbose=1,steps=358/32)

y_pred=np.argmax(pred,axis=1)



y_pred
y_true= test_generator.classes



cm= confusion_matrix(y_true,y_pred)

cm

#{'fire-accident': 0, 'flood': 1, 'non-disaster': 2}
from sklearn.metrics import classification_report

print(classification_report(y_true,y_pred))

from sklearn.metrics import accuracy_score

print("Densenet without earthquake",accuracy_score(y_true,y_pred ))

y_true_1=[]

for i in y_true:

    if i!=0:

        y_true_1.append(1)

    else:

        y_true_1.append(0)

y_pred_1=[]

for i in y_pred:

    if i!=0:

        y_pred_1.append(1)

    else:

        y_pred_1.append(0)

        

    



print(classification_report(y_true_1,y_pred_1))
