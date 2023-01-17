import pandas as pd

import numpy as np

from PIL import Image

from tqdm import tqdm



from keras.preprocessing.image import ImageDataGenerator #keras的数据增强类

from keras.layers import Conv2D, MaxPool2D, Dropout, BatchNormalization, Dense, Activation, GlobalAveragePooling2D

from keras.models import Model,Sequential

from keras.regularizers import l2
#导入数据，数据增强，划分训练集和验证集

def load_data(data=None, batch_size=32, mode='categorical'):

    if data is None:

        data = pd.read_csv('../input/train.csv')

    data['has_cactus'] = data['has_cactus'].astype('str')

    

    gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True, validation_split=0.1)

    

    trainGen = gen.flow_from_dataframe(data, directory='../input/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),

                                      class_mode=mode, batch_size=batch_size, shuffle=True, subset='training')

    validGen = gen.flow_from_dataframe(data, directory='../input/train/train', x_col='id', y_col='has_cactus', has_ext=True, target_size=(32, 32),

                                      class_mode=mode, batch_size=batch_size, shuffle=True, subset='validation')

    return trainGen, validGen

    
def base_model():

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



def train_model():

    batch_size = 32

    trainGen, validGen = load_data(batch_size=batch_size)

    model = base_model()

    

    opt = Adam(1e-3)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1, min_lr=1e-5, verbose=1)]

    model.fit_generator(trainGen, steps_per_epoch=4922, epochs=3, validation_data=validGen, validation_steps=493, shuffle=True, callbacks=cbs)

    

    return model
def predict_model():

    test_data = pd.read_csv('../input/sample_submission.csv')

    pred = np.empty((test_data.shape[0],))

    for n in tqdm(range(test_data.shape[0])):

        data = np.array(Image.open('../input/test/test/'+test_data.id[n]))

        data = data.astype(np.float32) / 255.

        pred[n] = model.predict(data.reshape((1, 32, 32, 3)))[0][1]

        

    test_data['has_cactus'] = pred

    test_data.to_csv('sample_submission.csv', index=False)
model = train_model()
predict_model()