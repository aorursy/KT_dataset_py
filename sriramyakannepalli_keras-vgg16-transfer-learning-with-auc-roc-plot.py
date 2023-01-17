# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import os,cv2

import json

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.models import Sequential, Model 

from keras.preprocessing.image import ImageDataGenerator

from keras import applications

from tqdm import tqdm, tqdm_notebook

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications import VGG16

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint

from sklearn.metrics import classification_report, confusion_matrix

from itertools import product
%load_ext autoreload

%autoreload 2



%matplotlib inline
PATH = "../input"

!ls {PATH}
train_dir = "../input/train/train"

test_dir = "../input/test/test"
df_train = pd.read_csv(f'{PATH}/train.csv',low_memory=False)

df_test = pd.read_csv(f'{PATH}/sample_submission.csv',low_memory=False)

df_train.has_cactus= df_train.has_cactus.astype(str)
print('Shape of Training data: {}'.format(df_train.shape))

print('Features: {}'.format(df_train.columns))
df_train.tail()
plt.figure(figsize = (6,5))

sns.set(style="darkgrid")

ax = sns.countplot(x = 'has_cactus',hue='has_cactus',data = df_train)

plt.xticks(rotation='vertical')

plt.xlabel('Count of Each Category', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
df_train['has_cactus'].value_counts()
im = cv2.imread("../input/train/train/01e30c0ba6e91343a12d2126fcafc0dd.jpg")

plt.imshow(im)
train_datagen = ImageDataGenerator(rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        fill_mode='nearest')
valid_datagen = ImageDataGenerator(rescale=1/.255)
batch_size= 32
train_generator = train_datagen.flow_from_dataframe(df_train[:15000], directory=train_dir, x_col='id', y_col='has_cactus', 

                    target_size=(150, 150), color_mode='rgb', classes=None, class_mode='binary',

                    batch_size=batch_size, 

                    shuffle=True, seed=None, 

                    save_to_dir=None, save_prefix='', save_format='png', 

                    subset=None, interpolation='nearest', drop_duplicates=True)
valid_generator = valid_datagen.flow_from_dataframe(df_train[15000:], directory=train_dir, x_col='id', y_col='has_cactus', 

                    target_size=(150, 150), color_mode='rgb', classes=None, 

                    class_mode='binary', batch_size=batch_size, 

                    shuffle=True, seed=None, 

                    save_to_dir=None, save_prefix='', save_format='png', 

                    subset=None, interpolation='nearest', drop_duplicates=True)
vgg16_net = VGG16(weights='imagenet', 

                  include_top=False, 

                  input_shape=(150, 150, 3))
vgg16_net.trainable = False

vgg16_net.summary()
model1 = Sequential()

model1.add(vgg16_net)

model1.add(Flatten())

model1.add(Dense(256))

model1.add(Activation('relu'))

model1.add(Dropout(0.5))

model1.add(Dense(1))

model1.add(Activation('sigmoid'))
filepath = "best_model.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

call_backs_list = [checkpoint]
from sklearn import metrics

import tensorflow as tf

from keras import backend as K



def auc(y_true, y_pred):

    auc = tf.metrics.auc(y_true, y_pred)[1]

    K.get_session().run(tf.local_variables_initializer())

    return auc
model1.compile(loss='binary_crossentropy',

              optimizer=Adam(lr=1e-5),

              metrics=['accuracy',auc])
batch_size = 32
%%time

history = model1.fit_generator(train_generator,validation_data = valid_generator,validation_steps=800,

                              epochs=100,

                              steps_per_epoch=2000 // batch_size,callbacks=call_backs_list,

                              verbose=2)
acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

AUC = history.history['auc']

val_AUC = history.history['val_auc']



epochs = range(1, len(acc) + 1)



plt.plot(epochs, acc, 'g', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()



plt.plot(epochs, loss, 'g', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()



plt.plot(epochs, AUC, 'g', label='AUC')

plt.plot(epochs, val_AUC, 'b', label='Val_AUC')

plt.xlabel('No.of epochs')

plt.ylabel('AUC')

plt.title('Training and validation AUC')

plt.legend()

plt.figure()

model1.load_weights("best_model.hdf5")

model1.compile(loss='binary_crossentropy',

              optimizer=Adam(lr=1e-5),

              metrics=['accuracy',auc])
un_test_img=[]

count=0

for i in os.listdir("../input/test/test/"):

    un_test_img.append(i)

    count+=1

un_test_image=[]

for i in tqdm(range(count)):

    img = image.load_img('../input/test/test/'+un_test_img[i], target_size=(150,150,3), grayscale=False)

    img = image.img_to_array(img)

    img = img/255

    un_test_image.append(img)

un_test_img_array = np.array(un_test_image)
len(un_test_img)
output = model1.predict_classes(un_test_img_array)
submission_save = pd.DataFrame()

submission_save['id'] = un_test_img

submission_save['has_cactus'] = output

submission_save.to_csv('submission.csv', header=True, index=False)