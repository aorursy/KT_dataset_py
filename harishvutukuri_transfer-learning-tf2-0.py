import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing import image

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input



from glob import glob



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



import os, sys

import warnings

warnings.filterwarnings('ignore')
!mkdir /kaggle/working/data

!mkdir /kaggle/working/data/train

!mkdir /kaggle/working/data/test

!mkdir /kaggle/working/data/train/food

!mkdir /kaggle/working/data/train/nonfood

!mkdir /kaggle/working/data/test/food

!mkdir /kaggle/working/data/test/nonfood
!cp /kaggle/input/food5k/Food-5K/training/0*.jpg /kaggle/working/data/train/nonfood

!cp /kaggle/input/food5k/Food-5K/training/1*.jpg /kaggle/working/data/train/food

!cp /kaggle/input/food5k/Food-5K/validation/0*.jpg /kaggle/working/data/test/nonfood

!cp /kaggle/input/food5k/Food-5K/validation/1*.jpg /kaggle/working/data/test/food
train_path = '/kaggle/working/data/train'

valid_path = '/kaggle/working/data/test'
IMAGE_SIZE = [200,200]
# number of files

train_image_files = glob(train_path + '/*/*.jpg')

valid_image_files = glob(valid_path + '/*/*.jpg')
# number of classes

folders = glob(train_path + '/*')

folders
# random image

plt.imshow(image.load_img(np.random.choice(train_image_files)));
ptm = VGG16(

    input_shape = IMAGE_SIZE + [3],

    weights = 'imagenet',

    include_top = False

)
# freezing VGG16 Model weights

ptm.trainable = False
# map data into feature vectors



K = len(folders) # no. of classes

x = Flatten()(ptm.output)

x = Dense(K, activation='softmax')(x)
model = Model(inputs=ptm.input, outputs=x)
# ImageDataGenerator

gen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.2,

    horizontal_flip=True,

    preprocessing_function=preprocess_input

)
# Generator



batch_size = 128



train_generator = gen.flow_from_directory(

    train_path,

    shuffle=True,

    target_size=IMAGE_SIZE,

    batch_size=batch_size

)



valid_generator = gen.flow_from_directory(

    valid_path,

    shuffle=False,

    target_size=IMAGE_SIZE,

    batch_size=batch_size

)
model.compile(

    loss='categorical_crossentropy',  # as generator yields one-hit encoded results

    optimizer='adam',

    metrics=['accuracy']

)
r = model.fit(

    train_generator,

    validation_data=valid_generator,

    epochs=10,

    steps_per_epoch=int(np.ceil(len(train_image_files)/batch_size)),

    validation_steps=int(np.ceil(len(valid_image_files)/batch_size))

)
plt.plot(r.history['loss'], label='train loss')

plt.plot(r.history['val_loss'], label='valid loss')

plt.legend();
plt.plot(r.history['accuracy'], label='train accuracy')

plt.plot(r.history['val_accuracy'], label='valid accuracy')

plt.legend();
x = Flatten()(ptm.output)



model = Model(inputs=ptm.input, outputs=x)



gen = ImageDataGenerator(preprocessing_function=preprocess_input)



batch_size = 128



train_generator = gen.flow_from_directory(

    train_path,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    class_mode='binary'

)



valid_generator = gen.flow_from_directory(

    valid_path,

    target_size=IMAGE_SIZE,

    batch_size=batch_size,

    class_mode='binary'

)
Ntrain = len(train_image_files)

Nvalid = len(valid_image_files)



# figuring output size

feat = model.predict(np.random.random([1] + IMAGE_SIZE + [3]))

D = feat.shape[1]



X_train = np.zeros((Ntrain, D))

Y_train = np.zeros(Ntrain)

X_valid = np.zeros((Nvalid, D))

Y_valid = np.zeros(Nvalid)
i = 0

for x, y in train_generator:

    features = model.predict(x)

    sz = len(y)

    X_train[i:i+sz] = features

    Y_train[i:i+sz] = y

    

    i += sz

    

    if i >=Ntrain:

        break
i = 0

for x, y in valid_generator:

    features = model.predict(x)

    sz = len(y)

    X_valid[i:i+sz] = features

    Y_valid[i:i+sz] = y

    

    i += sz

    

    if i >=Nvalid:

        break
X_train.max(), X_train.min()
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train2 = sc.fit_transform(X_train)

X_valid2 = sc.transform(X_valid)
# Sklearn Logistic Regression



from sklearn.linear_model import LogisticRegression



logr = LogisticRegression()

logr.fit(X_train2, Y_train)

print(f"Train Score: {logr.score(X_train2, Y_train)}, Valid Score: {logr.score(X_valid2, Y_valid)}")
# Tensorflow Logistic Regression



i = Input(shape=(D,))

x = Dense(1, activation='sigmoid')(i)



linearmodel = Model(i,x)



linearmodel.compile(loss='binary_crossentropy',

                   optimizer='adam',

                   metrics=['accuracy'])



r = linearmodel.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=batch_size, epochs=10)
plt.plot(r.history['loss'], label='train loss')

plt.plot(r.history['val_loss'], label='valid loss')

plt.legend();
plt.plot(r.history['accuracy'], label='train accuracy')

plt.plot(r.history['val_accuracy'], label='valid accuracy')

plt.legend();
!rm -r /kaggle/working/*