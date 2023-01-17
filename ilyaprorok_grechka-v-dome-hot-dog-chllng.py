import os

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from matplotlib.image import imread
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
font = {
    'family': 'serif',
    'color':  'darkred',
    'weight': 'bold',
    'size': 22,
}
SEED = 257

TRAIN_DIR = '../input/hotdogs-spbu/train/train/'
TEST_DIR = '../input/hotdogs-spbu/test/test/'
categories = ['hot dog', 'not hot dog']
X, y = [], []

for category in categories:
    category_dir = os.path.join(TRAIN_DIR, category)
    for image_path in os.listdir(category_dir):
        X.append(imread(os.path.join(category_dir, image_path)))
        y.append(category)
len(X), len(y)
X[0].shape
y = [1 if x == 'hot dog' else 0 for x in y]
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.25, random_state=SEED)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)
width = 100
height = 100
epochs = 80
dropout_rate = 0.5
batch_size = 50
input_shape = (height, width, 3)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(rotation_range=0.2, horizontal_flip=True)
train_iterator = datagen.flow(X_train, y_train, batch_size=32)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
base_inc = InceptionV3(weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)
model = models.Sequential()

model.add(base_inc)
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax'))

for layer in base_inc.layers:
  layer.trainable = False
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# on colab we had epochs=epochs, but kaggle will run this for too long, so we put epochs=10 as an illustration
model.fit_generator(train_iterator, epochs=10,verbose=1,validation_data=(X_test, y_test),steps_per_epoch=len(X_train)//batch_size)
roc_auc_score(y_test, model.predict(X_test))
base_inc2 = InceptionV3(weights='../input/inceptionv3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=input_shape)
model3 = models.Sequential()

model3.add(base_inc2)
model3.add(GlobalAveragePooling2D())
model3.add(Dense(1024, activation='relu'))
model3.add(Dropout(dropout_rate))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(512, activation='relu'))
model3.add(Dense(2, activation='softmax'))
for layer in (base_inc2.layers)[:-12]:
  layer.trainable = False
for layer in base_inc2.layers:
    print(layer, layer.trainable)
model3.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
datagen2 = ImageDataGenerator(featurewise_center=True,
                              featurewise_std_normalization=True,
                              rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True
                              )
train_iterator2 = datagen2.flow(X_train, y_train, batch_size=batch_size)
# again, less number of epochs
model3.fit_generator(train_iterator2, epochs=20,verbose=1,validation_data=(X_test, y_test),steps_per_epoch=len(X_train)//70)
roc_auc_score(y_test, model3.predict(X_test))
leaderboard_X = []
leaderboard_filenames = []
for image_path in os.listdir(TEST_DIR):
    leaderboard_X.append(imread(os.path.join(TEST_DIR, image_path)))
    leaderboard_filenames.append(image_path)
print(type(leaderboard_X))
print(type(X_test))
leaderboard_X = np.asarray(leaderboard_X)
leaderboard_X.shape
leaderboard_filenames.index('00ac06736a410ef1cddcd03a62feea5934077f9507a8992ac67c851e93a41496.png')
submission2 = pd.DataFrame(
    {
        'image_id': leaderboard_filenames, 
        'image_hot_dog_probability': leadeboard_predictions_2
    }
)
submission.head(661)
submission2.to_csv('submit(3).csv', index=False)