# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

os.listdir('../input/aptos2019-blindness-detection/')

# Any results you write to the current directory are saved as output.
BASE_DIR = '../input/aptos2019-blindness-detection'

TRAIN_IMG_DIR = '../input/aptos2019-blindness-detection/train_images'

TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"

BATCH_SIZE = 8

EPOCHES = 30

IMAGE_SIZE = 512
train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

train_df['diagnosis'] = train_df['diagnosis'].astype('str')

train_df['id_code'] = train_df['id_code'].astype(str)+'.png'

train_df.head(10)
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

test_df['id_code'] = test_df['id_code'].astype(str)+'.png'

test_df.head(10)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rescale = 1./255,

    horizontal_flip=True,

    vertical_flip=True,

    zoom_range=0.2,

    validation_split = 0.2

)

train_gen = datagen.flow_from_dataframe(

    dataframe = train_df,

    directory = TRAIN_IMG_DIR,

    class_mode ='categorical',

    x_col = 'id_code',

    y_col = 'diagnosis',

    batch_size = BATCH_SIZE,

    target_size = (IMAGE_SIZE,IMAGE_SIZE),

    subset = 'training',

    shuffle = True,

)

test_gen = datagen.flow_from_dataframe(

    dataframe = train_df,

    directory = TRAIN_IMG_DIR,

    class_mode= 'categorical',

    x_col = 'id_code',

    y_col = 'diagnosis',

    batch_size = BATCH_SIZE,

    target_size = (IMAGE_SIZE,IMAGE_SIZE),

    subset = 'validation',

    shuffle = True,

)

# y_train = train_df['diagnosis']

# from keras.utils import np_utils

# y_train = np_utils.to_categorical(y_train)

# num_classes = y_train.shape[1]

# print(y_train.shape)

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)

encoded_diagnosis = onehot_encoder.fit_transform(train_df[['diagnosis']])

print(encoded_diagnosis)
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout, GaussianNoise, GaussianDropout

from keras.layers import Flatten, BatchNormalization

from keras.layers.convolutional import Conv2D, SeparableConv2D

from keras.constraints import maxnorm

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

from keras import regularizers, optimizers

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.densenet import DenseNet201

from keras.layers import LeakyReLU
def build_model():

    model = Sequential()

    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(512,512,3)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(5, activation='sigmoid'))

    

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])

    return model



def build_inception():

    model = Sequential()

    model.add(InceptionResNetV2(include_top=False, weights=None, input_shape=(512,512,3)))

    model.add(Flatten())

    model.add(Dense(256))

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dropout(0.5))

    model.add(Dense(5, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])

    return model



def build_densent():

    model = Sequential()

    model.add(DenseNet201(include_top=False, weights=None, input_shape=(512,512,3), pooling=None))

    model.add(Flatten())

    model.add(Dense(256))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(128))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))

    model.add(Dense(64))

    model.add(LeakyReLU(alpha=0.2))

    model.add(Dropout(0.5))

    model.add(Dense(5, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])

    return model
model = build_inception()
model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop= EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 20)

# model_checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)
history = model.fit_generator(generator=train_gen,              

                                    steps_per_epoch=len(train_gen)//BATCH_SIZE,

                                    validation_data=test_gen,                    

                                    validation_steps=len(test_gen)//BATCH_SIZE,

                                    epochs=50,

                                    callbacks = [early_stop],

                                    use_multiprocessing = True,

                                    verbose=1)
test_loss, test_acc = model.evaluate_generator(

    test_gen,

    steps=len(test_gen)

)



print("Test accuracy : {}".format(test_acc * 100))

print("Test loss/error : {}".format(test_loss))
submission_df = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

#submission_df['diagnosis'] = submission_df['diagnosis'].astype('str')

submission_df['id_code'] = submission_df['id_code'].astype(str)+'.png'

print(submission_df)
submission_datagen=ImageDataGenerator(rescale=1./255)

submission_gen=submission_datagen.flow_from_dataframe(

    dataframe=submission_df,

    directory=TEST_IMG_DIR,

    x_col="id_code",    

    batch_size=BATCH_SIZE,

    shuffle=False,

    class_mode=None, 

    target_size=(IMAGE_SIZE,IMAGE_SIZE)

)
predictions=model.predict_generator(submission_gen, steps = len(submission_gen))
max_probability = np.argmax(predictions,axis = 1) 
submission_df['id_code'] = submission_df['id_code'].str.replace('.png','')

submission_df['diagnosis'] = max_probability

submission_df.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')

submission.head
submission.columns