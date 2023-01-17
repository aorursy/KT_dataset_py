# General libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# File paths to get Kaggle data

input_path = '../input/'

train_path = input_path + 'train/train/'

test_path = input_path + 'test/test/'



# Load data

train_df = pd.read_csv(input_path + 'train.csv')

sample = pd.read_csv(input_path + 'sample_submission.csv')



# Get ids and labels

train_id = train_df['id']

labels = train_df['has_cactus']

test_id = sample['id']
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(train_id, labels, test_size=0.2)
def fetch_images(ids, filepath):

    # Array to load images into

    arr = []

    for img_id in ids:

        img = plt.imread(filepath + img_id)

        arr.append(img)

        

    # Turn into numpy array and normalize pixel values

    arr = np.array(arr).astype('float32')

    arr = arr / 255

    return arr
# Redefine sets to contain images and not ids

x_train = fetch_images(ids=x_train, filepath=train_path)

x_val = fetch_images(ids=x_val, filepath=train_path)

test = fetch_images(ids=test_id, filepath=test_path)



# Get dimensions of each image

img_dim = x_train.shape[1:]
fig, ax = plt.subplots(nrows=2, ncols=3)

ax = ax.ravel()

plt.tight_layout(pad=0.2, h_pad=2)



for i in range(6):

    ax[i].imshow(x_train[i])

    ax[i].set_title('has_cactus = {}'.format(y_train.iloc[i]))
# Layers for the full model

from keras.models import Model

from keras.layers import Input, Dense, Flatten, Dropout, LeakyReLU, Activation

from keras.layers.normalization import BatchNormalization



# Pre-trained model

from keras.applications.densenet import DenseNet201
# Hyperparameters

batch_size = 64

epochs = 30

steps = x_train.shape[0] // batch_size
# Inputs

inputs = Input(shape=img_dim)



# DenseNet

densenet201 = DenseNet201(weights='imagenet', include_top=False)(inputs)



# Our FC layer

flat1 = Flatten()(densenet201)

dense1 = Dense(units=256, use_bias=True)(flat1)

batchnorm1 = BatchNormalization()(dense1)

act1 = Activation(activation='relu')(batchnorm1)

drop1 = Dropout(rate=0.5)(act1)



# Output

out = Dense(units=1, activation='sigmoid')(drop1)



# Create Model

model = Model(inputs=inputs, outputs=out)

model.compile(optimizer='adam', loss='binary_crossentropy')
from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=2, mode='max')



img_aug = ImageDataGenerator(rotation_range=20, vertical_flip=True, horizontal_flip=True)

img_aug.fit(x_train)
# Show architecture of model

from keras.utils import plot_model

print(model.summary())

plot_model(model, to_file='densenet201_model.png')
model.fit_generator(img_aug.flow(x_train, y_train, batch_size=batch_size), 

                    steps_per_epoch=steps, epochs=epochs, 

                    validation_data=(x_val, y_val), callbacks=[reduce_lr], 

                    verbose=2)
test_pred = model.predict(test, verbose=2)
sample['has_cactus'] = test_pred

sample.to_csv('densenet_model.csv', index=False)