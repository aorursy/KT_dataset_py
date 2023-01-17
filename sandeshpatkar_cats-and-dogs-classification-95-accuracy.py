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

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import random

import os

import time

import csv
print(os.listdir("../input"))
from sklearn.model_selection import train_test_split



#keras / tensorflow

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.layers import MaxPooling2D, BatchNormalization, Conv2D, Dropout, Flatten, Dense

from keras.callbacks import ModelCheckpoint

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import EarlyStopping



from keras.models import load_model

import shutil

def make_directory(dir_path):

    if os.path.exists(dir_path):

        shutil.rmtree(dir_path)

    os.makedirs(dir_path)
data_path = '../output/dogs-vs-cats'
file_test = '../input/dogs-vs-cats/test1.zip'

file_train = '../input/dogs-vs-cats/train.zip'
with zipfile.ZipFile('../input/dogs-vs-cats/train.zip', 'r') as zip_ref:

    zip_ref.extractall(data_path)



# extract test data

with zipfile.ZipFile('../input/dogs-vs-cats/test1.zip', 'r') as zip_ref:

    zip_ref.extractall(data_path)
train_path = os.path.sep.join([data_path, 'train/'])

test_path = os.path.sep.join([data_path, 'test1/'])

train_files = os.listdir(train_path)

test_files = os.listdir(test_path)

filelist = os.listdir(train_path)
categories = []

for filename in filelist:

    category = filename.split('.')[0]

    if category == 'dog':

        categories.append(1)

    else:

        categories.append(0)



df = pd.DataFrame({

    'filename': filelist,

    'category': categories

})
df.head()
df.tail()
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 

train_path
sample = random.choice(filelist)

img_sample = load_img(train_path+sample)

plt.imshow(img_sample)
#splitting the data in train and validation



train, val = train_test_split(df, test_size = 0.2, random_state = 1)
train = train.reset_index(drop = True)

val = val.reset_index(drop = True)
train_datagen = ImageDataGenerator(rotation_range = 10,

                                   width_shift_range = 0.2,

                                   height_shift_range = 0.1,

                                   shear_range = 0.1,

                                   zoom_range = 0.1,

                                   rescale = 1./255)
img_res = (200, 200)

batch_size = 64
val_datagen = ImageDataGenerator(rescale = 1./255)
train_gen = train_datagen.flow_from_dataframe(train, directory = train_path,

                                              x_col = 'filename',

                                              y_col = 'category',

                                              target_size = img_res,

                                              class_mode = 'binary',

                                              seed = 1,

                                              batch_size = batch_size)
val_gen = val_datagen.flow_from_dataframe(val, directory = train_path,

                                          class_mode = 'binary',

                                          x_col = 'filename',

                                          y_col = 'category',

                                          target_size = img_res,

                                          seed = 1,

                                          batch_size =batch_size

                                          )
model = tf.keras.Sequential([

                             Conv2D(32, (3,3), activation = 'relu', input_shape = (200, 200, 3),

                                    kernel_initializer = tf.keras.initializers.HeUniform()),

                             BatchNormalization(),

                             MaxPooling2D(),

                             Dropout(0.2),



                             Conv2D(64, (3,3), activation = 'relu',

                                    kernel_initializer = tf.keras.initializers.HeUniform(), kernel_regularizer = tf.keras.regularizers.l2(l2=0.01)),

                             BatchNormalization(),

                             MaxPooling2D(),

                             Dropout(0.3),



                             Conv2D(128, (3,3), activation = 'relu', kernel_initializer = tf.keras.initializers.HeUniform(), kernel_regularizer = tf.keras.regularizers.l2(l2=0.01,)),

                             BatchNormalization(),

                             MaxPooling2D(),

                             Dropout(0.3),



                             Conv2D(128, (3,3), activation = 'relu', kernel_initializer= tf.keras.initializers.HeUniform(), kernel_regularizer = tf.keras.regularizers.l2(l2=0.01)),

                             BatchNormalization(),

                             MaxPooling2D(),

                             Dropout(0.3),



                             Conv2D(256, (3,3), activation = 'relu', kernel_initializer= tf.keras.initializers.HeUniform()),

                             BatchNormalization(),

                             MaxPooling2D(),

                             Dropout(0.5),



                             Flatten(),

                             Dense(1, activation= 'sigmoid')





])
optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.0001, momentum= 0.9)
model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
#callbacks

callbacks_list = []



#checkpoint all

# checkpoint_path = '/content/gdrive/My Drive/dataset/checkpoints'

# best = '/best'

# all = '/all'



# checkpoint_all = ModelCheckpoint(checkpoint_path+all, monitor = 'loss', verbose = 1,

#                              mode = 'min')

# callbacks_list.append(checkpoint_all)



# #checkpoint_best

# checkpoint_best = ModelCheckpoint(checkpoint_path+best, monitor = 'loss', verbose = 1, save_best_only = True,

#                                   mode = 'auto', save_freq = 4, )



# callbacks_list.append(checkpoint_best)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,

                              patience=2, min_lr=0.000001)

callbacks_list.append(reduce_lr)





early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1)

callbacks_list.append(early_stop)

callbacks_list
e = 50




history = model.fit(train_gen,

                    epochs = e,

                    validation_data = val_gen,

                    steps_per_epoch = len(train_gen),

                    validation_steps = len(val_gen),

                    callbacks = callbacks_list

                    )





model.save('model.h5')
def summary_plot(history):

  acc = history.history['accuracy']

  val_acc = history.history['val_accuracy']



  loss = history.history['loss']

  val_loss = history.history['val_loss']



  plt.figure(figsize=(10,10))

  plt.subplot(211)

  plt.title('Loss')

  plt.plot(loss, color = 'red', label = 'loss')

  plt.plot(loss, val_loss, color = 'blue', label = 'val_loss')

  plt.legend()

  



  plt.subplot(212)

  plt.title('Accuracy')

  plt.plot(acc, color = 'red', label = 'acc')

  plt.plot(val_acc, color = 'blue', label = 'val_acc')

  plt.legend()
summary_plot(history)
#test data preparation



test_files = os.listdir(test_path)

df_test = pd.DataFrame({

    'filename': test_files

})

samples = df_test.shape[0]
test_datagen = ImageDataGenerator(rescale = 1./255)



test_gen = test_datagen.flow_from_dataframe(df_test,

                                            test_path,

                                            x_col = 'filename',

                                            y_col = None,

                                            class_mode = None,

                                            target_size = img_res, 

                                            batch_size = batch_size,

                                            shuffle = False)

predict = model.predict(test_gen, steps = np.ceil(samples/batch_size), verbose = 1)
predict
prediction = [1 if y > 0.5 else 0 for y in predict]
test_df = df_test.copy()
test_df.head()
test_df['category'] = prediction
test_df.head()
test_df['category'].unique()
label_map = dict((v,k) for k,v in train_gen.class_indices.items())

test_df['category'] = test_df['category'].replace(label_map)
test_df['category'] = test_df['category'].replace({ 'dog': 1, 'cat': 0 })

test_df['category'].value_counts().plot.bar()

start = 36

end = start + 18

sample_test = test_df[start:end]

sample_test.reset_index(inplace = True)

plt.figure(figsize=(12, 24))

for index, row in sample_test.iterrows():

    filename = row['filename']

    category = row['category']

    img = load_img(test_path+filename, target_size=img_res)

    plt.subplot(6, 3, index+1)

    plt.imshow(img)

    plt.xlabel(filename + '(' + "{}".format(category) + ')' )

plt.tight_layout()

plt.show()
#submission



submission_df = test_df.copy()

submission_df['id'] = submission_df['filename'].str.split('.').str[0]

submission_df['label'] = submission_df['category']

submission_df.drop(['filename', 'category'], axis=1, inplace=True)

submission_df.to_csv('submission.csv', index=False)