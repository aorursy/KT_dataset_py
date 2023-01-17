import os

import sys

import numpy as np

import pandas as pd

from keras.preprocessing import image

from sklearn.preprocessing import LabelEncoder

from keras.utils.np_utils import to_categorical

from keras.models import Model

from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input
data_root = "../input/"

img_folder = data_root + 'images/'

train_data = pd.read_csv(data_root + 'train.csv')

train_ID = train_data['id']

train_Y = train_data['species']

test_data = pd.read_csv(data_root + 'test.csv')

test_ID = test_data['id']



le = LabelEncoder()

train_y = le.fit_transform(train_Y)
def resize_img(img, max_dim=96):

    large_axis = max((0, 1), key=lambda x: img.size[x])

    scalar = max_dim / float(img.size[large_axis])

    resized = img.resize(

        (int(img.size[0] * scalar), int(img.size[1] * scalar)))

    return resized





def load_image_data(id_list, max_dim=96, center=True):

    X = np.empty((len(id_list), max_dim, max_dim, 1))

    for i, idnum in enumerate(id_list):

        x = image.load_img(

            (img_folder + str(idnum) + '.jpg'), grayscale=True)

        x = image.img_to_array(resize_img(x, max_dim=max_dim))

        height = x.shape[0]

        width = x.shape[1]

        if center:

            h1 = int((max_dim - height) / 2)

            h2 = h1 + height

            w1 = int((max_dim - width) / 2)

            w2 = w1 + width

        else:

            h1, w1 = 0, 0

            h2, w2 = (height, width)

        X[i, h1:h2, w1:w2, :] = x

    return np.around(X / 255.0)
def AlexNex(input_layer):

    conv_1 = Convolution2D(96, 11, 11, activation='relu', input_shape=(

        96, 96, 1), border_mode='same', name='conv1')(input_layer)

    max_pool_1 = MaxPooling2D((3, 3), strides=(2, 2))(conv_1)



    conv_2 = Convolution2D(256, 5, 5, border_mode='same',

                           activation='relu')(max_pool_1)

    max_pool_2 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)



    conv_3 = Convolution2D(384, 3, 3, border_mode='same',

                           activation='relu')(max_pool_2)

    conv_4 = Convolution2D(384, 3, 3, border_mode='same',

                           activation='relu')(conv_3)



    conv_5 = Convolution2D(256, 3, 3, border_mode='same',

                           activation='relu')(conv_4)

    max_pool_5 = MaxPooling2D((3, 3), strides=(2, 2))(conv_5)



    flat = Flatten()(max_pool_5)

    dense_1 = Dense(4096, init='glorot_normal', activation='relu')(flat)

    drop_1 = Dropout(0.5)(dense_1)



    dense_2 = Dense(4096, init='glorot_normal', activation='relu')(drop_1)

    drop_2 = Dropout(0.5)(dense_2)



    output_layer = Dense(99, activation='softmax')(drop_2)



    model = Model(input_layer, output_layer)

    return model
def NaiveCovNet(input_layer):

    x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1),

                      border_mode='same')(input_layer)

    x = (Activation('relu'))(x)

    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)



    # Now through the second convolutional layer

    x = (Convolution2D(32, 5, 5, border_mode='same'))(x)

    x = (Activation('relu'))(x)

    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)



    # Flatten our array

    x = Flatten()(x)

    dense_1 = Dense(1024, init='glorot_normal', activation='relu')(x)

    drop_1 = Dropout(0.5)(dense_1)



    dense_2 = Dense(99, init='glorot_normal', activation='relu')(drop_1)

    drop_2 = Dropout(0.5)(dense_2)



    output_layer = Dense(99, activation='softmax')(drop_2)

    model = Model(input_layer, output_layer)

    return model
input_layer = Input(shape=(96, 96, 1), name='image')

model = NaiveCovNet(input_layer)

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])



trian_X = load_image_data(train_ID)

train_y = to_categorical(train_y)



history = model.fit(trian_X, train_y, nb_epoch=10, batch_size=128)

#f_model = './model'

#json_string = model.to_json()

#open(os.path.join(f_model, 'model.json'), 'w').write(json_string)

#print('save weights')

#model.save_weights(os.path.join(f_model, 'model_weights.hdf5'))
#from keras.models import model_from_json

#f_model = './model'

#model_filename = 'cnn_model.json'

#weights_filename = 'cnn_model_weights.hdf5'

#json_string = open(os.path.join(f_model, model_filename)).read()

#model = model_from_json(json_string)



#model.summary()



#model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#model.load_weights(os.path.join(f_model,weights_filename))
X_test = load_image_data(test_ID)

CNN_pred = model.predict(X_test)

LABELS = sorted(pd.read_csv(os.path.join(data_root, 'train.csv')).species.unique())

save_File = pd.DataFrame(CNN_pred,index=test_ID,columns=LABELS)

save_File.to_csv("submission.csv", index=True)