# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



#prepara os dados

train_path = "../input/train/train"

test_path = "../input/test/test"



label_frame = pd.read_csv('../input/train.csv')

test_frame = pd.read_csv('../input/sample_submission.csv')

x_train = []

x_test = []

y_train = np.array(label_frame['has_cactus'])

#load images

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

for fname in label_frame['id']:

    image_path = os.path.join(train_path , fname)

    pil_image = image.load_img(image_path, target_size=(32, 32, 3))

    np_image = image.img_to_array(pil_image)

    x_train.append(np_image)

for fname in test_frame['id']:

    image_path = os.path.join(test_path,fname)

    pil_image = image.load_img(image_path,target_size = (32,32,3))

    np_image = image.img_to_array(pil_image)

    x_test.append(np_image)

#transfoma para array

x_train = np.array(x_train)

x_train = x_train.astype('float32')/255

x_test = np.array(x_test)

x_test = x_test.astype('float32')/255

print(x_train.shape)

print(x_test.shape)
augmentations = ImageDataGenerator(

    vertical_flip=True,

    horizontal_flip=True,

    zoom_range=0.1)



augmentations.fit(x_train)
#build model

from keras.applications import VGG16

from keras import models

from keras import layers

from keras import optimizers

from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau

def get_model():

    base = VGG16(include_top = False,weights = 'imagenet',input_shape = (32,32,3))

    base.trainable = True

    base.summary()

    set_trainable = False

    for layer in base.layers:

        if layer.name == 'block5_conv3':

            set_trainable = True

        if set_trainable:

            layer.trainable = True

        else:

            layer.trainable = False

    model = models.Sequential()

    model.add(base)

    model.add(layers.Flatten())

    

    model.add(layers.BatchNormalization())

    

    model.add(layers.Dense(256,activation = 'relu'))

    

    model.add(layers.BatchNormalization())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1,activation = 'sigmoid'))

    model.summary()

    model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['acc'])

    return model

    

model = get_model()

model.fit_generator(augmentations.flow(x_train,y_train),epochs = 100, steps_per_epoch=150)

y_predictions = model.predict(x_test)

result = pd.DataFrame({'id' : pd.read_csv('../input/sample_submission.csv')['id'],'has_cactus' : y_predictions.squeeze()})

result.to_csv("submissionMax.csv", index=False, columns=['id', 'has_cactus'])

print('submit successful')