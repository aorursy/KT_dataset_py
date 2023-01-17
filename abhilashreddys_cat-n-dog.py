import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,re, random,cv2
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,Dropout

from keras import backend as K

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
TRAIN_DIR_CAT = '../input/dogs-vs-cats/dataset/dataset/training_set/cats/'

train_img_cats = [TRAIN_DIR_CAT+i for i in os.listdir(TRAIN_DIR_CAT)] # use this for full dataset

TRAIN_DIR_DOG = '../input/dogs-vs-cats/dataset/dataset/training_set/dogs/'

train_img_dogs = [TRAIN_DIR_DOG+i for i in os.listdir(TRAIN_DIR_DOG)] # use this for full dataset
def make_data(list_img,enc):

    X=[]

    y=[]

    count = 0

    random.shuffle(list_img)

    for img in list_img:

       #X.append(Image.open(img).resize((inp_wid,inp_ht), Image.ANTIALIAS))

       X.append(cv2.resize(cv2.imread(img), (inp_wid,inp_ht), interpolation=cv2.INTER_CUBIC))

       y.append(enc)

    return X,y
inp_wid = 128

inp_ht = 128

batch_size = 16
X_cat,y_cat = make_data(train_img_cats,0)

X_dog,y_dog = make_data(train_img_dogs,1)

c = list(zip(X_cat+X_dog,y_cat+y_dog))

random.shuffle(c)

X,Y = list(zip(*c))

print(len(X))

print(len(Y))
X_train, X_val, Y_train, Y_val = train_test_split(X[0:4000],Y[0:4000], test_size=0.125, random_state=1)
n_train = len(X_train)

n_val = len(X_val)
model = Sequential()



# layer num 1

model.add(Conv2D(32,(3,3),input_shape=(inp_wid,inp_ht,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# layer num 2

model.add(Conv2D(64,(3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# layer num 3

model.add(Conv2D(128,(3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



# layer num 4

model.add(Conv2D(64,(3,3)))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.5))



model.add(Flatten())

model.add(Dense(128))

model.add(Activation('relu'))

model.add(BatchNormalization())

model.add(Dense(1))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)

val_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)
train_generator = train_datagen.flow(np.array(X_train), Y_train, batch_size=batch_size)

val_generator = val_datagen.flow(np.array(X_val), Y_val, batch_size=batch_size)
earlystop = EarlyStopping(patience=10)
lrr = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=2, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
callbacks = [earlystop,lrr]
history = model.fit_generator(

    train_generator, 

    steps_per_epoch=n_train // batch_size,

    epochs=32,

    validation_data=val_generator,

    validation_steps=n_val // batch_size,

    callbacks = callbacks

)
model.save_weights("model_weights.h5")

model.save('model_keras.h5')