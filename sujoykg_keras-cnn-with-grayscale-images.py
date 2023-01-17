import cv2

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import json

import os

from tqdm import tqdm, tqdm_notebook

from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D

from keras.models import Model

from keras.applications import VGG16

from keras.optimizers import Adam
train_dir = "../input/train/train/"

test_dir = "../input/test/test/"

train_df = pd.read_csv('../input/train.csv')

train_df.head()
im = cv2.imread("../input/train/train/01e30c0ba6e91343a12d2126fcafc0dd.jpg")

plt.imshow(im)
model = Sequential()



model.add(Conv2D(32, (5, 5), strides = (1, 1), name = 'conv0', input_shape = (32, 32, 1)))



model.add(BatchNormalization(axis = 3, name = 'bn0'))

model.add(Activation('relu'))



model.add(MaxPooling2D((2, 2), name='max_pool'))

model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))

model.add(Activation('relu'))

model.add(AveragePooling2D((3, 3), name='avg_pool'))



model.add(GlobalAveragePooling2D())

model.add(Dense(300, activation="relu", name='rl'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid', name='sm'))
model.compile(loss='binary_crossentropy',

              optimizer=Adam(lr=1e-5), 

              metrics=['accuracy'])
X_tr = []

Y_tr = []

imges = train_df['id'].values

for img_id in tqdm_notebook(imges):

    temp = cv2.imread(train_dir + img_id)

    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    temp = np.expand_dims(temp, axis=2)

    X_tr.append(temp)    

    Y_tr.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  

X_tr = np.asarray(X_tr)

X_tr = X_tr.astype('float32')

X_tr /= 255

Y_tr = np.asarray(Y_tr)
batch_size = 32

nb_epoch = 500
%%time

# Train model

history = model.fit(X_tr, Y_tr,

              batch_size=batch_size,

              epochs=nb_epoch,

              validation_split=0.2,

              shuffle=True,

              verbose=2)
with open('history.json', 'w') as f:

    json.dump(history.history, f)



history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot()

history_df[['acc', 'val_acc']].plot()
%%time

X_tst = []

Test_imgs = []

for img_id in tqdm_notebook(os.listdir(test_dir)):

    temp = cv2.imread(test_dir + img_id)

    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    temp = np.expand_dims(temp, axis=2)

    X_tst.append(temp)      

    Test_imgs.append(img_id)

X_tst = np.asarray(X_tst)

X_tst = X_tst.astype('float32')

X_tst /= 255
# Prediction

test_predictions = model.predict(X_tst)
sub_df = pd.DataFrame(test_predictions, columns=['has_cactus'])

sub_df['has_cactus'] = sub_df['has_cactus'].apply(lambda x: 1 if x > 0.7 else 0)
sub_df['id'] = ''

cols = sub_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

sub_df=sub_df[cols]
for i, img in enumerate(Test_imgs):

    sub_df.set_value(i,'id',img)
sub_df.head()
sub_df.to_csv('submission.csv',index=False)