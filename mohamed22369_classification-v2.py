# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import datetime as dt

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 16

import numpy as np

import os

import pandas as pd

import seaborn as sns

from keras.applications import xception

from keras.preprocessing import image

from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

from tqdm import tqdm
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

NUM_CATEGORIES = len(CATEGORIES)

SAMPLE_PER_CATEGORY = 200

SEED = 1987

data_dir = '../input/plant-seedlings-classification/'

train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')

sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))

sample_submission.head(2)
for category in CATEGORIES:

    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))
train = []

for category_id, category in enumerate(CATEGORIES):

    for file in os.listdir(os.path.join(train_dir, category)):

        train.append(['train/{}/{}'.format(category, file), category_id, category])

train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])

train.head(2)

train.shape
train = pd.concat([train[train['category'] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES])

train = train.sample(frac=1)

train.index = np.arange(len(train))

train.head(2)

train.shape
test = []

for file in os.listdir(test_dir):

    test.append(['test/{}'.format(file), file])

test = pd.DataFrame(test, columns=['filepath', 'file'])

test.head(2)

test.shape
def read_img(filepath, size):

    img = image.load_img(os.path.join(data_dir, filepath), target_size=size)

    img = image.img_to_array(img)

    return img
np.random.seed(seed=SEED)

rnd = np.random.random(len(train))

train_idx = rnd < 0.8

valid_idx = rnd >= 0.8

ytr = train.loc[train_idx, 'category_id'].values

yv = train.loc[valid_idx, 'category_id'].values

len(ytr), len(yv)
INPUT_SIZE = 299

POOLING = 'avg'

x_train = np.zeros((len(train), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, file in tqdm(enumerate(train['file'])):

    img = read_img(file, (INPUT_SIZE, INPUT_SIZE))

    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))

    x_train[i] = x

print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))
import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import cv2

from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np

from keras.utils import np_utils

from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model 

from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.layers import BatchNormalization

scale=299



model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (scale, scale, 3))

add_model=applications.Xception(weights = "imagenet", include_top=False, input_shape = (scale, scale, 3))



model = Sequential()

model.add(add_model)

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu',input_shape=(scale, scale, 3)))

model.add(BatchNormalization(axis=3))

model.add(Conv2D(20, kernel_size=(3, 3), activation='relu'))

model.add(BatchNormalization(axis=3))

model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

model.add(Dropout(0.2))





model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(64, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(12, activation='softmax'))





#compling and show model

model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model.summary()
gen = ImageDataGenerator(

            rotation_range=360.,

            width_shift_range=0.3,

            height_shift_range=0.3,

            zoom_range=0.3,

            horizontal_flip=True,

            vertical_flip=True)



train_data_dir = "../input/plant-seedlings-classification/train/"

train_generator = gen.flow_from_directory(

                        train_data_dir,

                        target_size = (scale, scale),

                        batch_size = 16, 

                        class_mode = "categorical")



checkpoint = ModelCheckpoint("Vgg19FreezeXception.h5", monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

early = EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=1, mode='auto')



STEP_SIZE_TRAIN=train_generator.n/16

model.fit_generator(

                    train_generator,

                    epochs = 50,

                    shuffle= True,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    callbacks = [checkpoint, early])
prediction = []

for filepath in test['filepath']:

    img = cv2.imread(os.path.join(data_dir,filepath))

    img = cv2.resize(img,(scale, scale))

    img = np.asarray(img)

    img = img.reshape(1,scale, scale,3)

    pred = model.predict(img)

    prediction.append(classes.get(pred.argmax(axis=-1)[0])) #Invert Mapping helps to map Label



test = test.drop(columns =['filepath'])

pred = pd.DataFrame({'species': prediction})

test =test.join(pred)

test.head()