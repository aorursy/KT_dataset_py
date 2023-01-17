import numpy as np

import pandas as pd

import keras

import keras

from keras.layers import Input, Dense, concatenate, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, GlobalMaxPool2D

from keras.models import Model

from matplotlib import pyplot as plt

from scipy.ndimage import rotate as rot

np.random.seed(2018) #Happy new year :)
train = pd.read_json("../input/train.json")
train.inc_angle = train.inc_angle.map(lambda x: 0.0 if x == 'na' else x)



def transform (df):

    images = []

    for i, row in df.iterrows():

        band_1 = np.array(row['band_1']).reshape(75,75)

        band_2 = np.array(row['band_2']).reshape(75,75)

        band_3 = band_1 + band_2

        

        band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())

        band_2_norm = (band_2 - band_2. mean()) / (band_2.max() - band_2.min())

        band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        

        images.append(np.dstack((band_1_norm, band_2_norm, band_3_norm)))

    

    return np.array(images)



def augment(images):

    image_mirror_lr = []

    image_mirror_ud = []

    image_rotate = []

    image_rotate2 = []



    for i in range(0,images.shape[0]):

        band_1 = images[i,:,:,0]

        band_2 = images[i,:,:,1]

        band_3 = images[i,:,:,2]

            

        # mirror left-right

        band_1_mirror_lr = np.flip(band_1, 0)

        band_2_mirror_lr = np.flip(band_2, 0)

        band_3_mirror_lr = np.flip(band_3, 0)

        image_mirror_lr.append(np.dstack((band_1_mirror_lr, band_2_mirror_lr, band_3_mirror_lr)))

        

        # mirror up-down

        band_1_mirror_ud = np.flip(band_1, 1)

        band_2_mirror_ud = np.flip(band_2, 1)

        band_3_mirror_ud = np.flip(band_3, 1)

        image_mirror_ud.append(np.dstack((band_1_mirror_ud, band_2_mirror_ud, band_3_mirror_ud)))

        

        #rotate 

        band_1_rotate = rot(band_1, 30, reshape=False)

        band_2_rotate = rot(band_2, 30, reshape=False)

        band_3_rotate = rot(band_3, 30, reshape=False)

        image_rotate.append(np.dstack((band_1_rotate, band_2_rotate, band_3_rotate)))

        

        #rotate 2

        band_1_rotate = rot(band_1, 60, reshape=False)

        band_2_rotate = rot(band_2, 60, reshape=False)

        band_3_rotate = rot(band_3, 60, reshape=False)

        image_rotate2.append(np.dstack((band_1_rotate, band_2_rotate, band_3_rotate)))

        

    mirrorlr = np.array(image_mirror_lr)

    mirrorud = np.array(image_mirror_ud)

    rotated = np.array(image_rotate)

    rotated2 = np.array(image_rotate2)

    images = np.concatenate((images, mirrorlr, mirrorud, rotated, rotated2))

    return images
train_X = transform(train)

train_y = np.array(train ['is_iceberg'])

train_angle = train.inc_angle.values
train_angle.shape, train_y.shape


train_X = augment(train_X)

train_y = np.concatenate((train_y,train_y, train_y, train_y, train_y))

train_angle = np.concatenate((train_angle,train_angle, train_angle, train_angle, train_angle))



train_X.shape, train_y.shape, train_angle.shape

Input_figure = Input(shape=(75,75,3), name='input1')

Input_angle = Input(shape=(1,), name = 'input2')



x = Conv2D(13, kernel_size=(3,3))(Input_figure)

x = BatchNormalization()(x)

x = Activation('elu')(x)

x = MaxPooling2D(pool_size=(2,2))(x)

x = Dropout(0.3)(x)



x = Conv2D(21, kernel_size=(3,3))(x)

x = BatchNormalization()(x)

x = Activation('elu')(x)

x = MaxPooling2D(pool_size=(2,2))(x)

x = Dropout(0.3)(x)



x = Conv2D(34, kernel_size=(3,3))(x)

x = BatchNormalization()(x)

x = Activation('elu')(x)

x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

x = Dropout(0.3)(x)





x = Conv2D(55, kernel_size=(3,3))(x)

x = BatchNormalization()(x)

x = Activation('elu')(x)

x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

x = Dropout(0.3)(x)





x = GlobalMaxPool2D()(x)



#concatenate x and angle 

x = concatenate([x, Input_angle])



x = Dense(89)(x)

x = BatchNormalization()(x)

x = Activation('elu')(x)

x = Dropout(0.3)(x)

x = Dense(89, activation='elu')(x)

out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[Input_figure, Input_angle], outputs=out)

model.summary()
opt = keras.optimizers.nadam()

model.compile(optimizer=opt,

              loss='binary_crossentropy', 

            )
batch_size = 64

early_stopping = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 0, mode= 'min')

model_filepath='.weights.best.hdf5'

checkpoint = keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [early_stopping, checkpoint]
# import tensorflow as tf

# with tf.device('/gpu:0'):

history = model.fit([train_X,train_angle.reshape(-1,1)], train_y.reshape(-1,1), batch_size = batch_size, epochs =50, verbose =1, validation_split = 0.2, 

          callbacks=callbacks_list)
model.load_weights(model_filepath)

#loaded_model.compile(loss='binary_crossentropy', optimizer = opt)
test_file = '../input/test.json'

test = pd.read_json(test_file)

test.inc_angle = test.inc_angle.replace('na',0)

test_X = transform(test)

print(test_X.shape)
y_hat = model.predict([test_X,test.inc_angle.values], verbose=1)

submission = pd.DataFrame({'id': test["id"], 'is_iceberg': y_hat.flatten()})

submission.to_csv('sub_keras_two_inputs.csv', index=False)
y_hat