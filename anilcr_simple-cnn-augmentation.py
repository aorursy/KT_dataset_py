# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import sklearn

import sklearn.model_selection

import scikitplot



import matplotlib.pyplot as plt

import tensorflow.keras as keras

import tensorflow as tf

# Any results you write to the current directory are saved as output.
from skimage.io import imread

from skimage.transform import rotate

train_df = pd.read_csv('../input/train.csv')
train_data = []

train_tgt = []



for idx, row in train_df.iterrows():

    img = imread('../input/train/train/' + row['id'])

    train_data.append(img)

    train_tgt.append(row['has_cactus'])



train_data = np.array(train_data)

train_tgt = np.array(train_tgt)
train_X, val_X, train_y, val_y = sklearn.model_selection.train_test_split(train_data, train_tgt, test_size=0.40)
m0 = train_X[:,:,:,0].mean()

m1 = train_X[:,:,:,1].mean()

m2 = train_X[:,:,:,2].mean()



def augmented(data: list, tgt: list):

    aug_data = []

    aug_tgt = []



    for i in range(data.shape[0]):

        img = data[i, :, :, :]



        # img[:,:,0] = (img[:,:,0] - m0)/128.

        # img[:,:,1] = (img[:,:,1] - m1)/128.

        # img[:,:,2] = (img[:,:,2] - m2)/128.

        img = img/255.

        

        aug_data.append(img)



        aug_data.append(rotate(img, 90.))

        aug_data.append(rotate(img, 180.))

        aug_data.append(rotate(img, 270.))



        aug_data.append(np.flip(img))

        aug_data.append(np.flip(img, axis=0))

        aug_data.append(np.flip(img, axis=1))

        

        aug_tgt.append(tgt[i])

        

        aug_tgt.append(tgt[i])

        aug_tgt.append(tgt[i])

        aug_tgt.append(tgt[i])

        

        aug_tgt.append(tgt[i])

        aug_tgt.append(tgt[i])

        aug_tgt.append(tgt[i])



    return np.array(aug_data), np.array(aug_tgt)



train_X, train_y = augmented(train_X, train_y)

val_X, val_y = augmented(val_X, val_y)
train_X.mean(axis=0).mean(axis=0).mean(axis=0)
val_X.mean(axis=0).mean(axis=0).mean(axis=0)
model = keras.Sequential([

    keras.layers.InputLayer(input_shape=(32, 32, 3), batch_size=None, name='input'),



    keras.layers.Conv2D(128, 1, 1, 'SAME', activation=keras.activations.relu, name='bottleneck1/conv1'),

    keras.layers.Conv2D(16, 3, 1, 'SAME', activation=keras.activations.relu, name='bottleneck1/conv2'),

    keras.layers.BatchNormalization(name='bottleneck1/bn1'),

    

    keras.layers.Conv2D(128, 1, 1, 'SAME', activation=keras.activations.relu, name='bottleneck2/conv1'),

    keras.layers.Conv2D(32, 3, 2, 'SAME', activation=keras.activations.relu, name='bottleneck2/conv2'),

    keras.layers.BatchNormalization(name='bottleneck2/bn1'),

    

    keras.layers.Conv2D(128, 1, 1, 'SAME', activation=keras.activations.relu, name='bottleneck3/conv1'),

    keras.layers.Conv2D(64, 3, 2, 'SAME', activation=keras.activations.relu, name='bottleneck3/conv2'),

    keras.layers.BatchNormalization(name='bottleneck3/bn1'),

    

    keras.layers.Flatten(name='flatten'),

    

    keras.layers.Dense(1, activation=keras.activations.sigmoid, name='prob')

], name='sequential0')
model.summary()
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,

                              patience=5, min_lr=1e-9, verbose=1)

early_stop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1e-6, patience=5, verbose=True)

model.compile(

    loss='binary_crossentropy',

    optimizer=keras.optimizers.SGD(lr=0.0001, decay=0),

    metrics=['accuracy'],

    verbose=False

)
history = model.fit(

    x=train_X,

    y=train_y,

    validation_data=(val_X, val_y),

    epochs=150, batch_size=50,

    callbacks=[reduce_lr]

)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])
pred_y = model.predict(train_X)
print('validation accuracy: {}'.format(sklearn.metrics.roc_auc_score(train_y, pred_y)))
dff = (train_y >0).flatten() != (pred_y > 0.75).flatten()

for idx in np.where(dff)[0]:

    plt.figure()

    print(val_y[idx], pred_y[idx])

    for i in range(4):

        I = val_X[idx+(2*i-1),:,:,:]

        plt.subplot(2, 2, i+1)

        plt.imshow(I)
train_y[0]
output_df = pd.read_csv('../input/sample_submission.csv')

test_X = []

for idx, row in output_df.iterrows():

    test_X.append(imread('../input/test/test/' + row['id'])/255.)
test_X = np.array(test_X)
test_y = model.predict(test_X)
output_df.head()
output_df['has_cactus'] = test_y
output_df.to_csv('submission.csv', index=False)
!head submission.csv
model.save_weights('weights.hdf')