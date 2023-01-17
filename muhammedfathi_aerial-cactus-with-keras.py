from PIL import Image

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator

import keras

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPooling2D,Activation, BatchNormalization, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam, SGD

from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau

from keras.regularizers import l2

import matplotlib.pyplot as plt
def load_data(dataframe=None, batch_size=16, mode='categorical'):

    if dataframe is None:

        dataframe = pd.read_csv('../input/train.csv')

    dataframe['has_cactus'] = dataframe['has_cactus'].apply(str)

    gen = ImageDataGenerator(rescale=1./255., validation_split=0.1, horizontal_flip=True, vertical_flip=True)

    

    trainGen = gen.flow_from_dataframe(dataframe, directory='../input/train/train/',  x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),

        class_mode=mode, batch_size=batch_size, shuffle=True, subset='validation')

    

    testGen = gen.flow_from_dataframe(dataframe, directory='../input/train/train/',  x_col='id',y_col='has_cactus', has_ext=True, target_size=(32, 32),

        class_mode=mode, batch_size=batch_size, shuffle=True, subset='validation')

    

    return trainGen, testGen
# load data

trainGen, valGen = load_data(batch_size=32)
# define model

def train_model():

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',

                     input_shape=(32, 32, 3)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))



    model.add(Conv2D(64, (3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.3))

    

    model.add(Conv2D(128, (3, 3), padding='same'))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Flatten())

#     model.add(GlobalAveragePooling2D())

    model.add(Dense(16))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.3))

    model.add(Dense(2))

    model.add(BatchNormalization())

    model.add(Activation('softmax'))

    

    return model
model = train_model()

# initiate Adam optimizer

opt = keras.optimizers.RMSprop(lr=0.0005, decay=1e-5)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5, verbose=1)]



model.fit_generator(trainGen, steps_per_epoch=4922, epochs=4, validation_data=valGen, 

    validation_steps=493, shuffle=True, callbacks=cbs)
test_set = pd.read_csv('../input/sample_submission.csv')

pred = np.empty((test_set.shape[0],))

for n in tqdm(range(test_set.shape[0])):

    data = np.array(Image.open('../input/test/test/'+test_set.id[n]))

    data = data.astype(np.float32) / 255.

    pred[n] = model.predict(data.reshape((1, 32, 32, 3)))[0][1]



test_set['has_cactus'] = pred

test_set.to_csv('sample_submission.csv', index=False)