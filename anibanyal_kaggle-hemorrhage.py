# !pip install pydicom

import os

import shutil

import pandas as pd

import numpy as np

import glob, pylab, pandas as pd

import pydicom

import matplotlib.pylab as plt

import seaborn as sns

import sklearn

from keras import layers, models
base = '/kaggle/input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection'          

if not os.path.exists('data'):

    os.mkdir('data')

shutil.copy(os.path.join(base, 'stage_2_train.csv'), 'data')
images_path = os.path.join(base, 'stage_2_train')

images = os.listdir(images_path)

images = images[:int(len(images) /50 )]
# fig = plt.figure(figsize=(15, 10))

# columns = 5; rows = 2

# imgs = [img for img in images]

# for i in range(1, columns*rows +1):

#     ds = pydicom.dcmread(os.path.join(images_path, imgs[i]))

#     fig.add_subplot(rows, columns, i)

#     plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

#     fig.add_subplot
df = pd.read_csv('data/stage_2_train.csv')
df.head(6)
# ID = df.loc[0]['ID'].split('_')

# label = ID[2]

# label
labels = df.Label.values[:len(images)*6 + 1 ]

# # each image has 6 labels ...so each element of ohe is the whole label of an image which contains 6 values

ohe = [labels[i: i + 6] for i in range(0, len(labels) - 6, 6)]
train_images = images[:int(len(images) * 0.8)]

test_images = images[int(len(images) * 0.8):]
X_train = [pydicom.dcmread(os.path.join(images_path, img)).pixel_array for img in train_images]
X_train = np.array(X_train)

print(X_train.shape)

print(X_train[0].shape)
y_train = ohe[:int(len(images) * 0.8)]
X_train = np.expand_dims(X_train, axis=4)

X_train.shape
# plt.title(y_train[500])

# plt.imshow(X_train[500], cmap=plt.cm.bone)

# plt.show()
#  X_train, y_train = sklearn.utils.shuffle(X_train, y_train)
input_shape = (512, 512, 1) #### X_train[0].shape -> (512, 512)

model = models.Sequential([

    layers.Conv2D(32, (3,3), input_shape=input_shape),

    layers.Conv2D(64, (3,3)),

    layers.Conv2D(64, (3,3)),

    layers.Conv2D(128, (3,3)),

    layers.Conv2D(128, (3,3)),

    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3)),

    layers.Conv2D(256, (3,3)),

    layers.Conv2D(256, (3,3)),

    layers.MaxPooling2D(),

    layers.Conv2D(512, (3,3)),

    layers.Conv2D(512, (3,3)),

    layers.Conv2D(1024, (3,3)),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(6, activation='sigmoid')    

])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

EPOCHS = 10

BATCH_SIZE = 32
# model.summary()
# history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

history = model.fit_gene(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)