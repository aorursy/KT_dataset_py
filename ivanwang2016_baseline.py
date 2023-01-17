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
from PIL import Image

from tqdm import tqdm



from keras.preprocessing.image import ImageDataGenerator



def load_data(dataframe=None, batch_size=16, mode='categorical'):

    if dataframe is None:

        dataframe = pd.read_csv('../input/train.csv')

    dataframe['has_cactus'] = dataframe['has_cactus'].apply(str)

    gen = ImageDataGenerator(rescale=1./255., validation_split=0.1, horizontal_flip=True, vertical_flip=True)



    trainGen = gen.flow_from_dataframe(dataframe, directory='../input/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),

        class_mode=mode, batch_size=batch_size, shuffle=True, subset='training')

    testGen = gen.flow_from_dataframe(dataframe, directory='../input/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),

        class_mode=mode, batch_size=batch_size, shuffle=True, subset='validation')

    

    return trainGen, testGen
from keras.layers import Conv2D, MaxPool2D, Dense, BatchNormalization, Activation, GlobalAveragePooling2D

from keras.models import Sequential, Model

from keras.regularizers import l2



def baseline_model():

    model = Sequential()



    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D())



    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D())



    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPool2D())



    model.add(GlobalAveragePooling2D())

    model.add(Dense(2, activation='softmax'))



    return model
from keras.optimizers import Adam, SGD

from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau





def train_baseline():

    batch_size = 32

    trainGen, valGen = load_data(batch_size=batch_size)

    model = baseline_model()



    opt = Adam(1e-3)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5, verbose=1)]



    model.fit_generator(trainGen, steps_per_epoch=4922, epochs=3, validation_data=valGen, 

        validation_steps=493, shuffle=True, callbacks=cbs)

        

    return model

    

def predict_baseline(model):

    testdf = pd.read_csv('../input/sample_submission.csv')

    pred = np.empty((testdf.shape[0],))

    for n in tqdm(range(testdf.shape[0])):

        data = np.array(Image.open('../input/test/test/'+testdf.id[n]))

        data = data.astype(np.float32) / 255.

        pred[n] = model.predict(data.reshape((1, 32, 32, 3)))[0][1]

    

    testdf['has_cactus'] = pred

    testdf.to_csv('sample_submission.csv', index=False)
model = train_baseline()

predict_baseline(model)