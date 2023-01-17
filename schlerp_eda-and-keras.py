import numpy as np
import pandas as pd
import os
import random
import PIL
import cv2
import matplotlib.pyplot as plt

import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.preprocessing.image import load_img
print('lets view some samples from the dataset!')

train_img_path = '../input/train/images'
train_masks_path = '../input/train/masks'

img_size = (100, 100)

plt.rcParams['figure.figsize'] = 12,6

print('loading images...')
for i in range(3):
    img_name = random.choice(os.listdir(train_img_path))
    img_path = os.path.join(train_img_path, img_name)
    img = np.array(load_img(img_path, grayscale=True, target_size=img_size)) / 255.
    mask_path = os.path.join(train_masks_path, img_name)
    img_mask = np.array(load_img(mask_path, grayscale=True, target_size=img_size)) / 255.
    masked_img = np.where(img_mask, 1, img)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title('Image')
    axs[1].imshow(img_mask)
    axs[1].set_title('Mask')
    axs[2].imshow(masked_img)
    axs[2].set_title('Masked Image')

print('done!')
print('building train image and mask dataset...')

train_img_dataset = []
train_mask_dataset = []
for img_name in os.listdir(train_img_path):
    img_path = os.path.join(train_img_path, img_name)
    mask_path = os.path.join(train_masks_path, img_name)
    train_img = cv2.imread(img_path, 0)
    train_img = cv2.resize(train_img, (100, 100))
    train_img = np.array(train_img).reshape((100, 100, 1))
    train_mask = cv2.imread(mask_path, 0)
    train_mask = cv2.resize(train_mask, (100, 100))
    train_mask = np.array(train_mask).reshape((100, 100, 1))
#     if np.sum(train_mask) > 0:
#         train_img_dataset.append(train_img)
#         train_mask_dataset.append(train_mask)
    train_img_dataset.append(train_img)
    train_mask_dataset.append(train_mask)

print('converting datasets to numpy arrays...')
train_img_dataset = np.array(train_img_dataset)
train_mask_dataset = np.array(train_mask_dataset)

print('done!')
print('building model...')
inputs = Input(shape=(100, 100, 1))
conv1 = Conv2D(32, 3, activation='elu', padding='same')(inputs)
conv1 = Conv2D(32, 3, activation='elu', padding='same')(conv1)
# 100, 100, 64

maxpool1 = MaxPool2D(2)(conv1)
conv2 = Conv2D(64, 3, activation='elu', padding='same')(maxpool1)
conv2 = Conv2D(64, 3, activation='elu', padding='same')(conv2)
# 50, 50, 128

maxpool2 = MaxPool2D(2)(conv2)
conv3 = Conv2D(128, 3, activation='elu', padding='same')(maxpool2)
conv3 = Conv2D(128, 3, activation='elu', padding='same')(conv3)
# 25, 25, 256

up1 = UpSampling2D(2)(conv3)
up1 = Concatenate()([conv2, up1])
up1 = Conv2D(128, 3, activation='elu', padding='same')(up1)
up1 = Conv2D(128, 3, activation='elu', padding='same')(up1)
# 50, 50, 128

up2 = UpSampling2D(2)(up1)
up2 = Concatenate()([conv1, up2])
up2 = Conv2D(64, 3, activation='elu', padding='same')(up2)
up2 = Conv2D(64, 3, activation='elu', padding='same')(up2)
# 100, 100, 64

outputs = Conv2D(32, 3, activation='elu', padding='same')(up2)
outputs = Conv2D(32, 3, activation='elu', padding='same')(outputs)
outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(outputs)
# 100, 100, 1

model = Model(inputs=[inputs], outputs=[outputs])

print('compiling model...')
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

print('model summary:')
print(model.summary())

print('done!')
print('fitting model...')
train_history = model.fit(train_img_dataset, 
                          train_mask_dataset, 
                          batch_size=128, 
                          epochs=1,
                          shuffle=True)
print('creating visualisation dataset...')

num_to_vis = 5
vis_img_dataset = train_img_dataset[0:num_to_vis]
vis_mask_dataset = train_mask_dataset[0:num_to_vis]

print('predicting masks...')
vis_mask_preds = model.predict(vis_img_dataset)

print('displaying predictions...')
for i in range(num_to_vis):
    img = vis_img_dataset[i].reshape(100, 100)
    mask = vis_mask_dataset[i].reshape(100, 100)
    pred = vis_mask_preds[i].reshape(100, 100)
    
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title('Image')
    axs[1].imshow(mask)
    axs[1].set_title('Mask')
    axs[2].imshow(pred)
    axs[2].set_title('Predicted Mask')

print('done!')
