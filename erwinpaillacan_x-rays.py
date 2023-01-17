import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras import datasets, layers, models

from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Model



import numpy as np

import SimpleITK as sitk

import cv2 as cv

import matplotlib.pyplot as plt

import os



from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

import shutil

plt.style.use('seaborn-darkgrid')

train_PATH = '/kaggle/input/chest-xray-pneumonia/chest_xray/train'

test_PATH  = '/kaggle/input/chest-xray-pneumonia/chest_xray/test/'

valid_PATH = '/kaggle/input/chest-xray-pneumonia/chest_xray/val/'



image_size = 180



labels = ['NORMAL', 'PNEUMONIA']



def load_data(directorio):

    images = []

    clases = []

    for label in labels:

        folder = os.path.join(directorio,label)

        clase = labels.index(label)

        for image in os.listdir(folder):

            read_image = cv.imread(os.path.join(folder,image),cv.IMREAD_GRAYSCALE)

            image_resized = cv.resize(read_image,(image_size,image_size))

            images.append(image_resized)

            clases.append(clase)

    return np.array(images), np.array(clases)

X_train, Y_train = load_data(train_PATH)

X_valid, Y_valid = load_data(valid_PATH)

X_test, Y_test   = load_data(test_PATH)





X_train = np.array(X_train).reshape(-1, image_size, image_size, 1)

X_valid = np.array(X_valid).reshape(-1, image_size, image_size, 1)

X_test = np.array(X_test).reshape(-1, image_size, image_size, 1)



X_train = X_train / 255

X_valid = X_valid /255

X_test = X_test / 255



X_test.shape
plt.figure(figsize=(6, 6))

for i in range(4):

    ax = plt.subplot(2, 2, i + 1)    

    plt.imshow(X_train[i*700].reshape(image_size,image_size))

    title = 'Normal' if Y_train[i*700]==0 else 'Pneumonia'

    plt.title(title)

    plt.axis("off")



plt.figure(figsize=(4, 4))

normal = sum(Y_train==0)

pneumonia = sum(Y_train==1)

[normal, pneumonia]

plt.bar(['normal', 'neumonia'], [normal, pneumonia])

plt.show()
model = models.Sequential()



model.add(layers.Conv2D(32, (7, 7), activation='relu', input_shape=(image_size,image_size,1)))

model.add(layers.BatchNormalization())

#model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(32, (7, 7), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(64, (7, 7), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, (7, 7), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))





model.add(layers.Conv2D(128, (5, 5), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, (5, 5), activation='relu'))

model.add(layers.BatchNormalization())

#model.add(layers.Conv2D(256, (3, 3), activation='relu'))

#model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))



model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Conv2D(256, (3, 3), activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))







model.add(layers.Flatten())





model.add(layers.Dense(512, activation='relu'))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(BatchNormalization(axis=-1,center=True,scale=False))

model.add(layers.Dropout(0.5))





model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()
model.compile(optimizer='adam',

              loss="binary_crossentropy",

              metrics=['accuracy'])





datagen = ImageDataGenerator(rotation_range=15,

                             shear_range=.2,

                              zoom_range=.2,

                              width_shift_range=.2,

                              height_shift_range=.2,

                              rescale=1,

                              horizontal_flip=False,

                              vertical_flip=False,

                              fill_mode='nearest'

                              )



history = model.fit(datagen.flow(X_train, Y_train, batch_size=64), epochs=50, validation_data=(X_valid, Y_valid))

test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=1)

print ('Acuuracy test data: '+str(test_acc*100)+' %')
pred = model.predict(X_test, batch_size = 8)

pred = np.where(pred>0.5,1,0)





# Get the confusion matrix

CM = confusion_matrix(Y_test, pred)



fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(8,8))

plt.title('Confusion matrix in test data')

plt.xticks(range(2), ['Normal','Pneumonia'], fontsize=10)

plt.yticks(range(2), ['Normal','Pneumonia'], fontsize=10)

plt.show()