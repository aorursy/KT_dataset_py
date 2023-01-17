import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import cv2

import imageio

import matplotlib.pyplot as plt

import warnings

import matplotlib.cbook

from PIL import Image



train_csv = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test_csv = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

dir0 = os.path.join('..', 'input/aptos2019-blindness-detection/')

dir1 = os.path.join(dir0, 'train_images/')



train_csv['path'] = train_csv['id_code'].map(lambda x: os.path.join(dir1, '{}.png'.format(x)))

#train_csv = train_csv.drop(columns=['id_code'])



#Se tiene el path de cada imagen correspondiente al id_code

print(train_csv)
#print(train_csv.get_value(0, "path"))

#pic=imageio.imread(train_csv.get_value(0, "path"))



#plt.figure(figsize=(6,6))

#plt.imshow(pic);

#plt.axis('off');



plot1 = train_csv['diagnosis'].hist(figsize = (10, 7))

plot1.set_title("Histograma de frecuencias de los niveles del RD")

plot1.set_xlabel("Niveles")

plot1.set_ylabel("Cantidad")

        
sizes = []

for i in range (3662):

    img = Image.open(train_csv.at[i, "path"])

    size = img.size

    if size not in sizes:

        sizes.append(size)



print("La cantidad de tamaños diferentes es: ", len(sizes))

print (sizes)

train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

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

    directory="../input/aptos2019-blindness-detection/train_images",

    x_col="id_code",

    y_col="diagnosis",

    batch_size=batch_size,

    shuffle=True,

    class_mode="categorical",

    target_size=(image_size,image_size),

    subset='training')



test_gen=datagen.flow_from_dataframe(

    dataframe=train_df,

    directory="../input/aptos2019-blindness-detection/train_images",

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
import numpy as np

import mnist

from keras.models import Sequential

from keras.layers import Dense



# Build the model.

model = Sequential([

  Dense(64, activation='relu', input_shape=[96,96,3]),

  Dense(64, activation='relu'),

  Dense(10, activation='softmax'),

])
model.compile(

  optimizer='adam',

  loss='categorical_crossentropy',

  metrics=['accuracy'],

)
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

print(model.summary())
history = model.fit_generator(generator=train_gen,              

                                    steps_per_epoch=len(train_gen),

                                    validation_data=test_gen,                    

                                    validation_steps=len(test_gen),

                                    epochs=3,

                                    use_multiprocessing = True,

                                    verbose=1)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()