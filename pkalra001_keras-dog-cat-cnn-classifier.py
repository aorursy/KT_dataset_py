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

        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile

with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:

    zip_ref.extractall('kaggle/working/Train')
with zipfile.ZipFile('../input/dogs-vs-cats/test1.zip','r') as zip_ref:

    zip_ref.extractall('kaggle/working/Test')
import numpy as np

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import random

import os

import tensorflow as tf

# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
dir = os.listdir('kaggle/working/Train/train')

print(dir[:6])

label = [1 if files.split('.')[0] == 'dog' else 0 for files in dir]

df = pd.DataFrame({'file' : dir, 'label' : label})

print(df.head())

print(df['label'].value_counts())
#printing random image

file = random.choice(dir)

image = load_img('kaggle/working/Train/train/' + file)

plt.imshow(image)
Img_width = 128

Img_length = 128

Img_size = (Img_width, Img_length)

Img_channels = 3
from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization



with tpu_strategy.scope():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (Img_width, Img_length, Img_channels)))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.25))

    

    model.add(Conv2D(64, (3, 3), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.25))

           

    model.add(Conv2D(128, (3, 3), activation = 'relu'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Dropout(0.25))    

          

    model.add(Flatten())

    model.add(Dense(512, activation = 'relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))



    model.add(Dense(2, activation = 'softmax'))

          

    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

    model.summary()



          
from keras.callbacks import EarlyStopping



earlystop = EarlyStopping(monitor = 'val_loss', patience = 10)
df.head()
#Converting labels back to cat/dogs to be used in imageGenerator

df['label'] = df['label'].replace({1 : 'dog', 0 : 'cat'})
train_df, validate_df = train_test_split(df, test_size = 0.2, random_state = 2)

train_df = train_df.reset_index(drop = True)

validate_df = validate_df.reset_index(drop = True) 
train_size = train_df.shape[0]

print(train_size)

validate_size = validate_df.shape[0]

print(validate_size)

batch_size = 128
train_img_gen = ImageDataGenerator(rotation_range = 20,

                                   rescale = 1./255,

                                   shear_range = 0.1, 

                                   zoom_range = 0.2, 

                                   horizontal_flip = True,

                                   width_shift_range = 0.1,

                                   height_shift_range = 0.1)

train_generator = train_img_gen.flow_from_dataframe(train_df, 

                                                    'kaggle/working/Train/train',

                                                    x_col = 'file',

                                                    y_col = 'label',

                                                    target_size = Img_size,

                                                   class_mode = 'categorical',

                                                   batch_size = batch_size)
validate_img_gen = ImageDataGenerator(rescale = 1./255)

validate_generator = validate_img_gen.flow_from_dataframe(validate_df, 

                                                          'kaggle/working/Train/train', 

                                                          x_col = 'file',

                                                          y_col = 'label',

                                                          target_size = Img_size,

                                                          class_mode = 'categorical',

                                                          batch_size = batch_size)

#Cheking the image generator

example_df = train_df.sample(n = 1).reset_index(drop = True)

example_generator  = train_img_gen.flow_from_dataframe(example_df, 'kaggle/working/Train/train', 

                                                       x_col = 'file', y_col = 'label',

                                                       target_size = Img_size,

                                                       class_mode = 'categorical')
plt.figure(figsize = (12, 12))

for i in range(0,12):

    plt.subplot(4, 3, i+1)

    for x, y in example_generator:

        img = x[0]

        plt.imshow(img)

        break

plt.show()
epochs = 28

fit = model.fit_generator(train_generator, 

                          epochs = epochs, 

                          validation_data = validate_generator, 

                          validation_steps = validate_size//batch_size,

                          steps_per_epoch = train_size//batch_size,

                         callbacks = [earlystop])
model.save_weights('cat-dogs-pk.h5')
test_file_dir = os.listdir('kaggle/working/Test/test1')

test_df = pd.DataFrame({

    'file' : test_file_dir

})

test_size = test_df.shape[0]
test_img_gen = ImageDataGenerator(rescale = 1./255)

test_generator = test_img_gen.flow_from_dataframe(test_df, 

                                                  'kaggle/working/Test/test1',

                                                  x_col = 'file',

                                                  y_col = None,

                                                  class_mode = None,

                                                  target_size = Img_size,

                                                  batch_size = batch_size,

                                                  shuffle = False)

                                                  
predict = model.predict_generator(test_generator, steps = np.ceil(test_size/batch_size))
test_df['cat'] = np.argmax(predict, axis = 1)
label_map = dict((v, k) for k,v in train_generator.class_indices.items())

test_df['cat'] = test_df['cat'].replace(label_map)
test_df['cat'] = test_df['cat'].replace({'dog' : 1, 'cat' : 0})
sub_df = test_df.copy()

sub_df['id'] = sub_df['file'].str.split('.').str[0]

sub_df['label'] = sub_df['cat']

sub_df.drop(['file', 'cat'], axis = 1, inplace = True)

sub_df.to_csv('submission.csv', index = False)