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

train_df = pd.read_csv('../input/train.csv')

train_df['diagnosis'] = train_df['diagnosis'].astype('str')

train_df['id_code'] = train_df['id_code'].astype(str)+'.png'
from keras.preprocessing.image import ImageDataGenerator



datagen=ImageDataGenerator(

    rescale=1./255, 

    validation_split=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



batch_size = 16

image_size = 96







train_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    target_size=(image_size,image_size),

    subset='training')



test_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical", 

    target_size=(image_size,image_size),

    subset='validation')
y_train = train_df['diagnosis']

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]
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
def build_model():

    # create model

    model = Sequential()

    #model.add(Reshape((x_train.shape[0],),))

    #model.add(GaussianDropout(0.3,input_shape=[96,96,3]))

    model.add(Conv2D(15, (3, 3), input_shape=[96,96,3], activation='relu'))

    model.add(GaussianDropout(0.3))

    model.add(Conv2D(30, (5, 5), activation='relu', kernel_constraint=maxnorm(3)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(50, (5, 5), activation='relu'))

    model.add(Conv2D(50, (7, 7), activation='relu'))

    

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(128, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.0001)

                   ,activity_regularizer=regularizers.l1(0.01)))

    # Compile model

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=0.0001, amsgrad=True), metrics=['accuracy'])

    return model
model = build_model()
from keras.callbacks import EarlyStopping, ModelCheckpoint

es= EarlyStopping(monitor='val_loss', mode ='min', verbose = 1, patience = 20)

mc = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only = True, mode ='min', verbose = 1)
model.fit_generator(generator=train_gen,              

                                    steps_per_epoch=len(train_gen),

                                    validation_data=test_gen,                    

                                    validation_steps=len(test_gen),

                                    epochs=50,

                                    callbacks = [es, mc], 

                                    use_multiprocessing = True,

                                    verbose=1)
from keras.models import load_model

model = load_model('model.h5')
submission_df = pd.read_csv('../input/sample_submission.csv')

#submission_df['diagnosis'] = submission_df['diagnosis'].astype('str')

submission_df['filename'] = submission_df['id_code'].astype(str)+'.png'
submission_datagen=ImageDataGenerator(rescale=1./255)

submission_gen=submission_datagen.flow_from_dataframe(

    dataframe=submission_df,

    directory="../input/test_images",

    x_col="filename",    

    batch_size=batch_size,

    shuffle=False,

    class_mode=None, 

    target_size=(image_size,image_size)

)
predictions=model.predict_generator(submission_gen, steps = len(submission_gen))
max_probability = np.argmax(predictions,axis=1) 
submission_df.drop(columns=['filename'], inplace= True)

submission_df['diagnosis'] = max_probability

submission_df.to_csv('submission.csv', index=False)