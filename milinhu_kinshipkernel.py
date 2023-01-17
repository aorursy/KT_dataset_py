

rootpath = '../input/'
import pandas as pd

import numpy as np





y_validation = np.load(rootpath+'y_validation.npy')

X_train = pd.read_csv(rootpath+'X_train.csv')
X_validation_input1 = np.load(rootpath+'X_validation_input1.npy')  

X_validation_input2 = np.load(rootpath+'X_validation_input2.npy')
def Imggen(path, batch_size, img_path):

    while 1:

        cnt = 0

        X_train_input1 = []

        X_train_input2 = []

        y=[]

        for index, row in img_path.iterrows():

            img1 = cv2.imread(path+row['img1'])

            img2 = cv2.imread(path+row['img2'])

            kinship = row['kinship']

            X_train_input1.append(img1)

            X_train_input2.append(img2)

            y.append(kinship)

            cnt += 1

            if cnt == batch_size:

                cnt = 0

                yield ([np.array(X_train_input1), np.array(X_train_input2)], np.array(y))

                X_train_input1 = []

                X_train_input2 = []

                y = []
from keras.layers import Dense, Dropout, Activation, Conv2D,GlobalMaxPool2D, Concatenate, Multiply, Dropout, Subtract

from keras.models import Sequential

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.applications import vgg16

from keras.applications import Xception

from sklearn.model_selection import train_test_split

from sklearn import metrics

from keras.optimizers import SGD

from keras import Input, Model

from keras.layers import GlobalAvgPool2D
def kinship_model():

    input1 = Input(shape=(224, 224, 3), name='input1')

    input2 = Input(shape=(224, 224, 3), name='input2')

    base_model = vgg16.VGG16(weights='imagenet',

                         include_top=False,

                         input_shape=(224,224,3))

    for x in base_model.layers[:-3]:

        x.trainable = True

    x1 = base_model(input1)

    x2 = base_model(input2)

    x1 = Concatenate(axis=-1)([GlobalMaxPool2D()(x1), GlobalAvgPool2D()(x1)])

    x2 = Concatenate(axis=-1)([GlobalMaxPool2D()(x2), GlobalAvgPool2D()(x2)])

    x3 = Subtract()([x1, x2])

    x3 = Multiply()([x3, x3])



    x1_ = Multiply()([x1, x1])

    x2_ = Multiply()([x2, x2])

    x4 = Subtract()([x1_, x2_])

    merged = Concatenate(axis=-1)([x4, x3])

    dense_mid = Dense(600, kernel_regularizer=None, kernel_initializer='glorot_uniform',

                  activity_regularizer= None, activation='relu')(merged)

    dense_mid = Dense(600, kernel_regularizer=None, kernel_initializer='glorot_uniform',

                  activity_regularizer=None, activation='relu')(dense_mid)

    dense_out = Dense(1, activation='sigmoid', name='dense_out')(dense_mid)

    model = Model(inputs=[input1, input2], outputs=[dense_out])

    return model 
from keras import optimizers

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import cv2

# file_path = "kinship_model.h5"

path = rootpath+'train/'+'train/'

model = kinship_model()

model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

b_size = 30

max_epochs = 10

print("Starting training ")

# checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

# callbacks_list = [checkpoint, reduce_on_plateau]

model.fit_generator(Imggen(path, b_size, X_train), steps_per_epoch=len(X_train)//b_size,

                    epochs = max_epochs, 

                    validation_data = ([X_validation_input1, X_validation_input2],y_validation), 

                    use_multiprocessing=True)

print("Training finished \n")
model.save('kinship_model.h5')