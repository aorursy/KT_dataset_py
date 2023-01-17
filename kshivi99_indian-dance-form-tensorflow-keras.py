# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        os.path.join(dirname, filename)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow import keras

import os

import random

from shutil import copyfile
train_dir = '/kaggle/input/identify-the-dance-form/train/'

test_dir = '/kaggle/input/identify-the-dance-form/test/'

train_csv = pd.read_csv(r'/kaggle/input/identify-the-dance-form/train.csv')

test_csv = pd.read_csv(r'/kaggle/input/identify-the-dance-form/test.csv')
train_csv.head()
os.mkdir(r'/kaggle/working/identify-dance-form')

os.mkdir(r'/kaggle/working/identify-dance-form/training')

os.mkdir(r'/kaggle/working/identify-dance-form/testing')

os.mkdir(r'/kaggle/working/identify-dance-form/source')

os.mkdir(r'/kaggle/working/identify-dance-form/training/manipuri')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/manipuri')

os.mkdir(r'/kaggle/working/identify-dance-form/source/manipuri')

os.mkdir(r'/kaggle/working/identify-dance-form/training/bharatanatyam')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/bharatanatyam')

os.mkdir(r'/kaggle/working/identify-dance-form/source/bharatanatyam')

os.mkdir(r'/kaggle/working/identify-dance-form/training/odissi')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/odissi')

os.mkdir(r'/kaggle/working/identify-dance-form/source/odissi')

os.mkdir(r'/kaggle/working/identify-dance-form/training/kathakali')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/kathakali')

os.mkdir(r'/kaggle/working/identify-dance-form/source/kathakali')

os.mkdir(r'/kaggle/working/identify-dance-form/training/kathak')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/kathak')

os.mkdir(r'/kaggle/working/identify-dance-form/source/kathak')

os.mkdir(r'/kaggle/working/identify-dance-form/training/sattriya')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/sattriya')

os.mkdir(r'/kaggle/working/identify-dance-form/source/sattriya')

os.mkdir(r'/kaggle/working/identify-dance-form/training/kuchipudi')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/kuchipudi')

os.mkdir(r'/kaggle/working/identify-dance-form/source/kuchipudi')

os.mkdir(r'/kaggle/working/identify-dance-form/training/mohiniyattam')

os.mkdir(r'/kaggle/working/identify-dance-form/testing/mohiniyattam')

os.mkdir(r'/kaggle/working/identify-dance-form/source/mohiniyattam')

os.mkdir('/kaggle/working/identify-dance-form/tests')

os.mkdir('/kaggle/working/identify-dance-form/tests/unknown')
files = []

for  r, d, f in os.walk(train_dir):

    for file in f:

        if '.jpg' in file:

            files.append(file)
import matplotlib.image as mpimg

plt.imshow(mpimg.imread(os.path.join(train_dir,files[0])))
for x in files:

    if (train_csv[train_csv['Image'] == x]['target'] == 'odissi').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/odissi/',x)

        copyfile(train_temp, final_train)

    elif (train_csv[train_csv['Image'] == x]['target'] == 'manipuri').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/manipuri/',x)

        copyfile(train_temp, final_train)

    elif (train_csv[train_csv['Image'] == x]['target'] == 'bharatanatyam').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/bharatanatyam/',x)

        copyfile(train_temp, final_train)

    elif (train_csv[train_csv['Image'] == x]['target'] == 'kathakali').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/kathakali/',x)

        copyfile(train_temp, final_train)

    elif (train_csv[train_csv['Image'] == x]['target'] == 'kathak').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/kathak/',x)

        copyfile(train_temp, final_train)

    elif (train_csv[train_csv['Image'] == x]['target'] == 'sattriya').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/sattriya/',x)

        copyfile(train_temp, final_train)

    elif (train_csv[train_csv['Image'] == x]['target'] == 'kuchipudi').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/kuchipudi/',x)

        copyfile(train_temp, final_train)

    elif (train_csv[train_csv['Image'] == x]['target'] == 'mohiniyattam').bool():

        train_temp = os.path.join(train_dir,x)

        final_train = os.path.join('/kaggle/working/identify-dance-form/source/mohiniyattam/',x)

        copyfile(train_temp, final_train)
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):

    shuffle=random.sample(os.listdir(SOURCE),len(os.listdir(SOURCE)))

    train_data_length=int(len(os.listdir(SOURCE))*SPLIT_SIZE)

    test_data_length=int(len(os.listdir(SOURCE))-train_data_length)

    train_data=shuffle[0:train_data_length]

    test_data=shuffle[-test_data_length:]

    for x in train_data:

        train_temp=os.path.join(SOURCE,x)

        final_train=os.path.join(TRAINING,x)

        copyfile(train_temp,final_train)

    for x in test_data:

        test_temp=os.path.join(SOURCE,x)

        final_test=os.path.join(TESTING,x)

        copyfile(test_temp,final_test)
bhatanatyam_source_dir = '/kaggle/working/identify-dance-form/source/bharatanatyam/'

bhatanatyam_training_dir = '/kaggle/working/identify-dance-form/training/bharatanatyam/'

bhatanatyam_testing_dir = '/kaggle/working/identify-dance-form/testing/bharatanatyam/'



kathak_source_dir = '/kaggle/working/identify-dance-form/source/kathak/'

kathak_training_dir = '/kaggle/working/identify-dance-form/training/kathak/'

kathak_testing_dir = '/kaggle/working/identify-dance-form/testing/kathak/'



kathakali_source_dir = '/kaggle/working/identify-dance-form/source/kathakali/'

kathakali_training_dir = '/kaggle/working/identify-dance-form/training/kathakali/'

kathakali_testing_dir = '/kaggle/working/identify-dance-form/testing/kathakali/'



kuchipudi_source_dir = '/kaggle/working/identify-dance-form/source/kuchipudi/'

kuchipudi_training_dir = '/kaggle/working/identify-dance-form/training/kuchipudi/'

kuchipudi_testing_dir = '/kaggle/working/identify-dance-form/testing/kuchipudi/'



manipuri_source_dir = '/kaggle/working/identify-dance-form/source/manipuri/'

manipuri_training_dir = '/kaggle/working/identify-dance-form/training/manipuri/'

manipuri_testing_dir = '/kaggle/working/identify-dance-form/testing/manipuri/'



mohiniyattam_source_dir = '/kaggle/working/identify-dance-form/source/mohiniyattam/'

mohiniyattam_training_dir = '/kaggle/working/identify-dance-form/training/mohiniyattam/'

mohiniyattam_testing_dir = '/kaggle/working/identify-dance-form/testing/mohiniyattam/'



odissi_source_dir = '/kaggle/working/identify-dance-form/source/odissi/'

odissi_training_dir = '/kaggle/working/identify-dance-form/training/odissi/'

odissi_testing_dir = '/kaggle/working/identify-dance-form/testing/odissi/'



sattriya_source_dir = '/kaggle/working/identify-dance-form/source/sattriya/'

sattriya_training_dir = '/kaggle/working/identify-dance-form/training/sattriya/'

sattriya_testing_dir = '/kaggle/working/identify-dance-form/testing/sattriya/'
split_size = 0.85

split_data(bhatanatyam_source_dir, bhatanatyam_training_dir, bhatanatyam_testing_dir, split_size)

split_data(sattriya_source_dir, sattriya_training_dir, sattriya_testing_dir, split_size)

split_data(odissi_source_dir, odissi_training_dir, odissi_testing_dir, split_size)

split_data(mohiniyattam_source_dir, mohiniyattam_training_dir, mohiniyattam_testing_dir, split_size)

split_data(manipuri_source_dir, manipuri_training_dir, manipuri_testing_dir, split_size)

split_data(kuchipudi_source_dir, kuchipudi_training_dir, kuchipudi_testing_dir, split_size)

split_data(kathakali_source_dir, kathakali_training_dir, kathakali_testing_dir, split_size)

split_data(kathak_source_dir, kathak_training_dir, kathak_testing_dir, split_size)
from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout



num_classes = 8



my_new_model = Sequential()

my_new_model.add(VGG16(include_top = False, pooling='avg', weights = 'imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))



# Indicate whether the first layer should be trained/changed or not.

my_new_model.layers[0].trainable = False
my_new_model.compile(optimizer='sgd', 

                     loss='categorical_crossentropy', 

                     metrics=['accuracy'])
from tensorflow.keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator



image_size = 224

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   horizontal_flip=True,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.2)



TRAINING_DIR = "/kaggle/working/identify-dance-form/training"

train_generator = data_generator.flow_from_directory(

                                        directory=TRAINING_DIR,

                                        target_size=(image_size, image_size),

                                        batch_size=10,

                                        class_mode='categorical')



VALIDATION_DIR = "/kaggle/working/identify-dance-form/testing"

validation_generator = data_generator.flow_from_directory(

                                        directory=VALIDATION_DIR,

                                        target_size=(image_size, image_size),

                                        class_mode='categorical')



# fit_stats below saves some statistics describing how model fitting went

# the key role of the following line is how it changes my_new_model by fitting to data

fit_stats = my_new_model.fit_generator(train_generator,steps_per_epoch=30,

                                       validation_data=validation_generator,

                                       validation_steps=1)
fil = []

for  r, d, f in os.walk(test_dir):

    for file in f:

        if '.jpg' in file:

            fil.append(file)
for x in fil:

    train_temp = os.path.join(test_dir,x)

    final_train = os.path.join('/kaggle/working/identify-dance-form/tests/unknown',x)

    copyfile(train_temp, final_train)

    

tests_dir = '/kaggle/working/identify-dance-form/tests'

test_generator = data_generator.flow_from_directory(tests_dir,

                                                  target_size = (224,224),

                                                  batch_size=32, class_mode='categorical')



img_list = []

for x in test_generator.filenames:

    x = x.split('/')[1]

    img_list.append(x)

    

predictions = my_new_model.predict_generator(test_generator)



predicted_clases = np.argmax(predictions,axis=-1)
train_generator.class_indices




data = {'Image': img_list, 'target': predicted_clases}



df = pd.DataFrame(data)

df.head()
df['target']= df['target'].map({0: 'bharatanatyam',

                                1: 'kathak',

                                2: 'kathakali',

                                3: 'kuchipudi',

                                4: 'manipuri',

                                5: 'mohiniyattam',

                                6: 'odissi',

                                7: 'sattriya'})
df.to_csv(r'submission_dance.csv', index = False)