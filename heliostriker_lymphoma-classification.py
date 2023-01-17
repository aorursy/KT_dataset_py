!pip install git+https://github.com/HelioStrike/pathomics-research-utils.git

from pathomics_research_utils.data_generators import ClassificationDataGenerator

from pathomics_research_utils.models import AlexNet

from pathomics_research_utils import utils

import tensorflow as tf

import tensorflow.keras.layers as layers

from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Model

import numpy as np

import random

import cv2

import os
#Hyperparams

lr = 1e-4

n_epochs = 8

num_classes = 3
!wget http://andrewjanowczyk.com/wp-static/lymphoma.tar.gz

!tar -zxvf lymphoma.tar.gz

!mkdir lymphoma

!mv CLL lymphoma

!mv FL lymphoma

!mv MCL lymphoma

os.listdir('.')

!rm -r lymphoma/FL/.DS_Store
image_height = 224

image_width = 224

batch_size=4
class ClassificationDataGenerator(Sequence):

    def __init__(self, images_dir=None, height=32, width=32, resize=False,

                 batch_size=32, shuffle=True, augmentation=None,

                 magnify=None, num_channels=3, num_classes=3, start_ratio=0, end_ratio=0.8):

        self.batch_size = batch_size

        self.num_channels = num_channels

        self.num_classes = num_classes

        self.shuffle = shuffle

        self.augmentation = augmentation



        self.shuffle = shuffle

        self.images = []

        self.image_labels = []

        self.dir_names = os.listdir(images_dir)

        for i in range(len(self.dir_names)):

            dir_path = os.path.join(images_dir, self.dir_names[i])

            fnames = os.listdir(dir_path)

            cur=0

            for fname in fnames:

                try:

                    img = os.path.join(dir_path, fname)

                    self.images.append(img)

                    self.image_labels += [i]

                    cur+=1

                except:

                    pass

        if self.shuffle:

            self.shuffle_data()

        self.image_labels = self.image_labels[int(start_ratio*len(self.images)):int(end_ratio*len(self.images))]

        self.images = self.images[int(start_ratio*len(self.images)):int(end_ratio*len(self.images))]

        self.height = height

        self.width = width

        self.resize = resize

        self.magnify = magnify

        self.len = len(self.images) // self.batch_size



    def __len__(self):

        return self.len



    def shuffle_data(self):

        a = list(zip(self.images, self.image_labels))

        random.shuffle(a)

        self.images, self.image_labels = zip(*a)



    def on_epoch_start(self):

        if self.shuffle:

            self.shuffle_data()



    def __getitem__(self, idx):

        X = np.empty((self.batch_size, self.height, self.width, self.num_channels))

        y = np.zeros((self.batch_size, self.num_classes))

        for i in range(self.batch_size):

            img = utils.read_image(self.images[idx*self.batch_size+i])

            if self.resize:

                img = cv2.resize(img, (self.height, self.width))

            X[i] = img

            y[i][self.image_labels[idx*self.batch_size+i]] = 1

            if self.augmentation:

                X[i] = self.augmentation(X[i])

        return X, y
data_generator = ClassificationDataGenerator(images_dir='lymphoma', height=image_height, width=image_width, resize=True, batch_size=batch_size, start_ratio=0, end_ratio=0.8)

val_generator = ClassificationDataGenerator(images_dir='lymphoma', height=image_height, width=image_width, resize=True, batch_size=batch_size, start_ratio=0.8, end_ratio=1)
def get_model(dropout=None):

    net = tf.keras.applications.ResNet50(weights='imagenet')

    if dropout is not None:

        drp = layers.Dropout(dropout)(net.output)

        y_pred = layers.Dense(num_classes, activation='softmax')(drp)        

    else:

        y_pred = layers.Dense(num_classes, activation='softmax')(net.output)

    return Model(inputs=net.input, outputs=y_pred)
model = get_model(0.2)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(

    generator=data_generator,

    validation_data=val_generator,

    epochs=n_epochs

)