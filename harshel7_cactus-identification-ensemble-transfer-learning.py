# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import os

import matplotlib.pyplot as plt

import seaborn as sns

import cv2

from tqdm import tqdm_notebook



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read in all our files

train_df = pd.read_csv('../input/train.csv')

train_images = '../input/train/*'

test_images = '../input/test/*'
train_df.head()
sns.set(style = 'darkgrid')

plt.figure(figsize = (12,10))

sns.countplot(train_df['has_cactus'])
#let's visualize some cactus images

IMAGES = os.path.join(train_images, "*")

all_images = glob.glob(IMAGES)
#visualize some images



plt.figure(figsize = (12,10))

plt.subplot(1, 3, 1)

plt.imshow(plt.imread(all_images[0]))

plt.xticks([])

plt.yticks([])



plt.subplot(1, 3, 2)

plt.imshow(plt.imread(all_images[10]))

plt.xticks([])

plt.yticks([])



plt.subplot(1, 3, 3)

plt.imshow(plt.imread(all_images[20]))

plt.xticks([])

plt.yticks([])
train_path = '../input/train/train/'

test_path = '../input/test/test/'
#let's get our image data and image labels toegether

#read in all the images

images_id = train_df['id'].values

X = [] #this list will contain all our images

for id_ in images_id:

    img = cv2.imread(train_path + id_)

    X.append(img)
#now let's get our labels

label_list = [] #will contain all our labels

for img_id in images_id:

    label_list.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])
#now we can convert our images list and the labels list into numpy array

X = np.array(X)

y = np.array(label_list)
print(f"THE SIZE OF OUR TRAINING DATA : {X.shape}")

print(f"THE SIZE OF OUR TRAINING LABELS : {y.shape}")
#let's do some preprocessing such as normalizing our data

X = X.astype('float32') / 255
#loading in and preprocessing the test data

X_test = []

test_images = []

for img_id in tqdm_notebook(os.listdir(test_path)):

    X_test.append(cv2.imread(test_path + img_id))     

    test_images.append(img_id)

X_test = np.array(X_test)

X_test = X_test.astype('float32') / 255
#import the required libraries

import keras

from keras.layers import Conv2D

from keras.layers import Dense

from keras.layers import BatchNormalization

from keras.layers import Flatten

from keras.layers import Activation

from keras.layers import Dropout

from keras.layers import MaxPooling2D

from keras.models import Sequential

from keras import optimizers

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras import backend as K
class CNN:

    def build(height, width, classes, channels):

        model = Sequential()

        inputShape = (height, width, channels)

        chanDim = -1

        

        if K.image_data_format() == 'channels_first':

            inputShape = (channels, height, width)

            chanDim = 1

        model.add(Conv2D(32, (3,3), padding = 'same', input_shape = inputShape))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Activation('relu'))

        model.add(Conv2D(32, (3,3), padding = 'same'))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Activation('relu'))

        model.add(MaxPooling2D(2,2))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(128, (3,3), padding = 'same', input_shape = inputShape))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Activation('relu'))

        model.add(Conv2D(128, (3,3), padding = 'same'))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Activation('relu'))

        model.add(MaxPooling2D(2,2))

        model.add(Dropout(0.25))

        

        model.add(Conv2D(256, (3,3), padding = 'same', input_shape = inputShape))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Activation('relu'))

        model.add(Conv2D(256, (3,3), padding = 'same'))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Activation('relu'))

        model.add(MaxPooling2D(2,2))

        model.add(Dropout(0.25))

        

        model.add(Flatten())

        

        model.add(Dense(128, activation = 'relu'))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Dropout(0.5))

        

        model.add(Dense(32, activation = 'relu'))

        model.add(BatchNormalization(axis = chanDim))

        model.add(Dropout(0.5))

        

        model.add(Dense(classes, activation = 'sigmoid'))

        

        return model
input_dim = X.shape[1:]

activation = 'relu'

classes = 1

height = 32

width = 32

channels = 3



history = dict() #dictionery to store the history of individual models for later visualization

prediction_scores = dict() #dictionery to store the predicted scores of individual models on the test dataset



#here we will be training the same model for a total of 10 times and will be considering the mean of the output values for predictions

for i in np.arange(0, 5):

    optim = optimizers.Adam(lr = 0.001)

    ensemble_model = CNN.build(height = height, width = width, classes = classes, channels = channels)

    ensemble_model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])

    print('TRAINING MODEL NO : {}'.format(i))

    H = ensemble_model.fit(X, y,

                           batch_size = 32,

                           epochs = 200,

                           verbose = 1)

    history[i] = H

    

    ensemble_model.save('MODEL_{}.model'.format(i))

    

    predictions = ensemble_model.predict(X_test, verbose = 1, batch_size = 32)

    prediction_scores[i] = predictions
from keras.applications.vgg16 import VGG16
vgg16 = VGG16(weights = 'imagenet', input_shape = (32, 32, 3), include_top = False)

vgg16.summary()
for layer in vgg16.layers:

    layer.trainable = False
vgg_model = Sequential()

vgg_model.add(vgg16)

vgg_model.add(Flatten())

vgg_model.add(Dense(256, activation = 'relu'))

vgg_model.add(BatchNormalization())

vgg_model.add(Dropout(0.5))

vgg_model.add(Dense(128, activation = 'relu'))

vgg_model.add(BatchNormalization())

vgg_model.add(Dropout(0.5))

vgg_model.add(Dense(1, activation = 'sigmoid'))



vgg_model.summary()
#compile the model

vgg_model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on our data

vgg_history = vgg_model.fit(X, y,

                            batch_size = 64,

                            epochs = 500,

                            verbose = 1) 
#making predictions on test dat

predictions_vgg = vgg_model.predict(X_test)
predictions_vgg.shape
from keras.applications.resnet50 import ResNet50
resnet = ResNet50(weights = 'imagenet', input_shape = (32, 32, 3), include_top = False)

resnet.summary()
for layer in resnet.layers:

    layer.trainable = False
resnet_model = Sequential()

resnet_model.add(resnet)

resnet_model.add(Flatten())

resnet_model.add(Dense(256, activation = 'relu'))

resnet_model.add(BatchNormalization())

resnet_model.add(Dropout(0.5))

resnet_model.add(Dense(128, activation = 'relu'))

resnet_model.add(BatchNormalization())

resnet_model.add(Dropout(0.5))

resnet_model.add(Dense(1, activation = 'sigmoid'))



resnet_model.summary()
#compile the model

resnet_model.compile(loss = 'binary_crossentropy', optimizer = optim, metrics = ['accuracy'])
#fit the model on our data

resnet_history = resnet_model.fit(X, y,

                                  batch_size = 64, 

                                  epochs = 500,

                                  verbose = 1) 
resnet_predictions = resnet_model.predict(X_test)
#making predictions

prediction = np.hstack([p.reshape(-1,1) for p in prediction_scores.values()]) #taking the scores of all the trained models

predictions_ensemble = np.mean(prediction, axis = 1)

print(predictions_ensemble.shape)
df_ensemble = pd.DataFrame(predictions_ensemble, columns = ['has_cactus'])

df_ensemble['has_cactus'] = df_ensemble['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)
df_ensemble['id'] = ''

cols = df_ensemble.columns.tolist()

cols = cols[-1:] + cols[:-1]

df_ensemble = df_ensemble[cols]



for i, img in enumerate(test_images):

    df_ensemble.set_value(i,'id',img)



#making submission

df_ensemble.to_csv('ensemble_submission.csv',index = False)
df_vgg = pd.DataFrame(predictions_vgg, columns = ['has_cactus'])

df_vgg['has_cactus'] = df_vgg['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)
df_vgg['id'] = ''

cols = df_vgg.columns.tolist()

cols = cols[-1:] + cols[:-1]

df_vgg = df_vgg[cols]



for i, img in enumerate(test_images):

    df_vgg.set_value(i,'id',img)



#making submission

df_vgg.to_csv('vgg_submission.csv',index = False)
df_vgg.head()
df_resnet = pd.DataFrame(resnet_predictions, columns = ['has_cactus'])

df_resnet['has_cactus'] = df_resnet['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)
df_resnet['id'] = ''

cols = df_resnet.columns.tolist()

cols = cols[-1:] + cols[:-1]

df_resnet = df_resnet[cols]



for i, img in enumerate(test_images):

    df_resnet.set_value(i,'id',img)



#making submission

df_resnet.to_csv('resnet_submission.csv',index = False)
df_vgg1 = pd.DataFrame(predictions_vgg, columns = ['has_cactus'])

df_ensemble1 = pd.DataFrame(predictions_ensemble, columns = ['has_cactus'])



df_t = 0.5 * df_vgg1['has_cactus'] + 0.5 * df_ensemble1['has_cactus']

df_t = pd.DataFrame(df_t, columns = ['has_cactus'])

df_t['has_cactus'] = df_t['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)



df_t['id'] = ''

cols = df_t.columns.tolist()

cols = cols[-1:] + cols[:-1]

df_t = df_t[cols]



for i, img in enumerate(test_images):

    df_t.set_value(i,'id',img)



#making submission

df_t.to_csv('vgg_ensemble_submission.csv',index = False)
df_vgg2 = pd.DataFrame(predictions_vgg, columns = ['has_cactus'])

df_ensemble2 = pd.DataFrame(predictions_ensemble, columns = ['has_cactus'])

df_resnet2 = pd.DataFrame(resnet_predictions, columns = ['has_cactus'])



df_t2 = 0.45 * df_vgg2['has_cactus'] + 0.45 * df_ensemble2['has_cactus'] + 0.10 * df_resnet2['has_cactus']

df_t2 = pd.DataFrame(df_t2, columns = ['has_cactus'])

df_t2['has_cactus'] = df_t2['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)



df_t2['id'] = ''

cols = df_t2.columns.tolist()

cols = cols[-1:] + cols[:-1]

df_t2 = df_t2[cols]



for i, img in enumerate(test_images):

    df_t2.set_value(i,'id',img)



#making submission

df_t2.to_csv('vgg_ensemble_resnet_submission.csv',index = False)