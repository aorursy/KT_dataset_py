import os

import cv2

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

import pandas as pd
def crop_image_from_gray(img,tol=7):

    if img.ndim ==2:

        mask = img>tol

        return img[np.ix_(mask.any(1),mask.any(0))]

    elif img.ndim==3:

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        mask = gray_img>tol

        

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]

        if (check_shape == 0): # image is too dark so that we crop out everything,

            return img # return original image

        else:

            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]

            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]

            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]

    #         print(img1.shape,img2.shape,img3.shape)

            img = np.stack([img1,img2,img3],axis=-1)

    #         print(img.shape)

        return img
path = ['/kaggle/input/glaucomadataset/Non Glaucoma', '/kaggle/input/glaucomadataset/Glaucoma']

images = []

labels = []

for n,i in enumerate(path):

    for j in tqdm(os.listdir(i)):

        img_path = os.path.join(i,j)

        img = cv2.imread(img_path)

        img = crop_image_from_gray(img,tol=7)

        img = cv2.resize(img, (224,224))

        images.append(img)

        labels.append(n)

images = np.array(images)/255

labels = np.array(labels)
plt.figure(figsize=(20,20))

for i in range(1,26):

    plt.subplot(5,5,i)

    n = np.random.randint(1022)

    plt.imshow(images[n])

    plt.title(labels[n])
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split



images, labels = shuffle(images, labels, random_state=32)

x_train, x_valid, y_train, y_valid = train_test_split(images, labels, test_size=0.15, random_state=44)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.15, random_state=40)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True, vertical_flip=True,

                            fill_mode='constant', cval=0.)

train_gen = datagen.flow(x_train, y_train, batch_size=32)
from keras.models import Sequential, model_from_json

from keras.layers import Conv2D, Dense, Dropout, BatchNormalization, Flatten, MaxPool2D, GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.applications.densenet import DenseNet121
tr = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
model = Sequential()

model.add(tr)

model.add(GlobalAveragePooling2D())

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

ck = ModelCheckpoint('glaucoma_weights.hdf5', monitor='val_loss', save_best_only=True, mode='auto', verbose=1)

re = ReduceLROnPlateau(monitor='val_loss', mode='auto', factor=0.2, patience=4, verbose=1)

model.summary()
history = model.fit_generator(train_gen, epochs=50, steps_per_epoch=800//32,

                              verbose=1, validation_data=(x_valid,y_valid),

                              callbacks=[ck,re])
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['accuracy', 'val_accuracy']].plot()
pred = model.evaluate(x_test, y_test)

print('Test Accuracy:', pred[1]*100)
model_json = model.to_json()

with open("glaucoma_model.json", "w") as json_file:

    json_file.write(model_json)