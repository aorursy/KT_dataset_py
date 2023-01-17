#load libraries for data manipulation and visualization

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

# warnings

import string

import warnings

warnings.filterwarnings('ignore')
# load the train and test data sets

train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print('Number of Training Samples = {}'.format(train.shape[0]))

print('Number of Test Samples = {}\n'.format(test.shape[0]))

print('Training X Shape = {}'.format(train.shape))

print('Training y Shape = {}\n'.format(train['label'].shape[0]))

print('Test X Shape = {}'.format(test.shape))

print('Test y Shape = {}\n'.format(test.shape[0]))

print('Index of Train Set:\n', train.columns)

print('Index of Test Set:\n', test.columns)
# check datatypes

train.info()
# sample of data

train.head()
lp = sb.countplot(train['label'])
# check for missing values

missing_train = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_train = missing_train[missing_train > 0] * 100

print("There are {} train features with  missing values :".format(missing_train.shape[0]))

missing_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_test= missing_test[missing_test > 0] * 100

print("There are {} test features with  missing values :".format(missing_test.shape[0]))

def show_image(train_image, train_label, index):

    image_shaped = train_image.values.reshape(28,28)

    plt.subplot(4, 6, index+1)

    plt.imshow(image_shaped, cmap=plt.cm.gray)

    plt.title(label)





plt.figure(figsize=(18, 8))

sample_image = train.sample(24).reset_index(drop=True)

for index, row in sample_image.iterrows():

    label = row['label']

    image_pixels = row.drop('label')

    show_image(image_pixels, label, index)

plt.tight_layout()
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical



# prepare training and test sets

X = train.drop(columns=['label']).values.reshape((train.shape[0],28,28,1))

X_train = X.astype('float32') / 255

y = to_categorical(train['label'])

X_test = test.values.reshape((test.shape[0],28,28,1))

X_test = X_test.astype('float32') / 255



# prepare training and validation sets

train_images, test_images, train_labels, test_labels = train_test_split(X, y,

                                                test_size=0.1, random_state=0)

train_images = train_images.astype('float32') / 255

test_images = test_images.astype('float32') / 255
# defining a small convnet

import tensorflow.keras.models as models

import tensorflow.keras.layers as layers

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))



# adding a classifier on top of the convnet

model.add(layers.Flatten())

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau



callbacks = [

    EarlyStopping(patience=10, verbose=1),

    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),

    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)

]
# model training 

model.compile(optimizer='rmsprop',

   loss='categorical_crossentropy',

   metrics=['acc'])

history = model.fit(train_images,

   train_labels,

   epochs=20,

   batch_size=64,

   callbacks=callbacks,

   validation_data=(test_images, test_labels))
predictions = model.predict(X_test)

results = np.argmax(predictions, axis = 1) 
plt.figure(figsize=(18, 8))

sample_test = test.head(24)

for index, image_pixels in sample_test.iterrows():

    label = results[index]

    show_image(image_pixels, label, index)

plt.tight_layout()
solution = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

solution['Label'] = results

solution.to_csv('solution.csv', index = False)