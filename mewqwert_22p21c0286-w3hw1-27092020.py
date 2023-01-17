import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import os
import tensorflow as tf

import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.regularizers import l2
import tempfile

from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.losses import BinaryCrossentropy

from tensorflow.keras import layers
from tensorflow.keras import Model
start = '/kaggle/input/super-ai-image-classification'
locate = {'train' : start+'/train/train/train.csv', 
          'img_train_folder' : start+'/train/train/images', 
          'val' : start+'/val/val/val.csv', 
          'img_val_folder' : start+('/val/val/images')+''}
img = {'train' : os.listdir(locate['img_train_folder']), 
       'val' : os.listdir(locate['img_val_folder'])}
train = pd.read_csv(locate['train'])
import hashlib
duplicates_f = {}
hash_key = {}
for i, file in enumerate(train.iloc[:, 0].values):
    f = open(locate['img_train_folder'] + '/' + file, 'rb')
    f_hash = hashlib.md5(f.read()).hexdigest()
    if f_hash not in hash_key:
        hash_key[f_hash] = i
    else:
        duplicates_f[file] = i
train_dup = train.copy()
for j in duplicates_f.values():
    train_dup.drop(j, inplace = True)
train_dup.reset_index(drop=True)
train.head()
x = train_dup['id'].values
y = train_dup['category'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.25,
      height_shift_range=0.25,
      zoom_range=0.3,
      horizontal_flip=True,
      brightness_range = [0.7, 1.3],
      channel_shift_range = 150.0,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)
from tensorflow.keras.preprocessing import image
d = pd.read_csv(os.getcwd()+'/drive/My Drive/test.csv')
target_sizes = (388,388)
x_im = []
xd = d.iloc[:, 0].values
yd = d.iloc[:, 1].values
for im in xd:
    temp = image.load_img(locate['img_val_folder'] + '/' + im, target_size=target_sizes, interpolation='nearest')
    my_img = np.array(image.img_to_array(temp))
    x_im.append(my_img)
x_train_img = []
x_test_img = []

for im in x_train:
    temp = image.load_img(locate['img_train_folder'] + '/' + im, target_size=target_sizes, interpolation='nearest')
    my_img = np.array(image.img_to_array(temp))
    x_train_img.append(my_img)
for im in x_test:
    temp = image.load_img(locate['img_train_folder'] + '/' + im, target_size=target_sizes, interpolation='nearest')
    my_img = np.array(image.img_to_array(temp))
    x_test_img.append(my_img)
x_train_img += x_im
y_train = np.concatenate((y_train, yd), axis =0)
x_train_img = np.array(x_train_img)
x_test_img = np.array(x_test_img)
y_train = np.array(y_train)
y_test = np.array(y_test)
densenet = tf.keras.applications.densenet.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (388,388, 3),
                                                     pooling = 'max')
# fit output
x = tf.keras.layers.Flatten()(densenet.output)
x = tf.keras.layers.Dropout(rate = 0.8)(x)
x = tf.keras.layers.Dense(3715, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.7)(x)
x = tf.keras.layers.Dense(2700, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.6)(x)
x = tf.keras.layers.Dense(300, activation='relu')(x)
x = tf.keras.layers.Dropout(rate = 0.5)(x)
x = tf.keras.layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(densenet.input, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(lr=0.0001),metrics=['accuracy'])
model.summary()
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train_img)
## fits the model on batches with real-time data augmentation:
history = model.fit(datagen.flow(x_train_img, y_train, batch_size = 32),
                    steps_per_epoch = len(x_train_img) / 60, epochs = 30, validation_data = (x_test_img, y_test))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_hat = model.predict(x_test_img)

from sklearn.metrics import f1_score
#y_hat_trnf = y_hat.round()
y_hat_trnf = y_hat.argmax(axis=1)

F_measure = f1_score(y_test, y_hat_trnf)
print("F1 :", F_measure)
df = img['val']
x_val = []
for im in df:
    temp = image.load_img(locate['img_val_folder'] + '/' + im, target_size=target_sizes, interpolation='nearest')
    my_img = np.array(image.img_to_array(temp))
    x_val.append(my_img)
x_val = np.array(x_val)
y_val = model.predict(x_val)
y_val = y_val.argmax(axis=1)
sub = pd.DataFrame({'id': df,
              'category':y_val})
sub.to_csv('val_final.csv', index = False)