# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pickle as pickle

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import random



import keras as k

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



import cv2

import datetime as dt

from tqdm import tqdm



from multiprocessing import Pool, cpu_count



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Import Model Specific packages

from keras.preprocessing.image import img_to_array, load_img



import keras as k

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPooling2D



# import packages for InceptionV3

from keras.applications.inception_v3 import InceptionV3

from keras.models import Model



from multiprocessing import Pool, cpu_count



# callback for saving models, early stopping

from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping



# for plotting model training history

import matplotlib.pyplot as plt



# os.chdir('C:/deep_learning') # This is where the input dataset is stored

# os.getcwd()

print("------Fan Fei's Imports Complete-----")
random_seed = 987654321

random.seed(random_seed)

np.random.seed(random_seed)

input_dim = 299



x_train0 = []

x_test0 = []

y_train0 = []



df_train = pd.read_csv('../input/train_v2.csv')[:256]        # just get the first 256 images



labels = df_train['tags'].str.get_dummies(sep=' ').columns



label_map = {l: i for i, l in enumerate(labels)}

inv_label_map = {i: l for l, i in label_map.items()}



for f, tags in tqdm(df_train.values, miniters=1000):

    img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1 

    x_train0.append(cv2.resize(img, (input_dim, input_dim)))

    y_train0.append(targets)

    

y_train0 = np.array(y_train0, np.uint8)

x_train0 = np.array(x_train0, np.float16) / 255.



print(x_train0.shape)

print(y_train0.shape)
df_test = pd.read_csv('../input/sample_submission_v2.csv')[:256]        # just get the first 256 images



for f, tags in tqdm(df_test.values, miniters=1000):

    img = cv2.imread('../input/test-jpg-v2/{}.jpg'.format(f))

    targets = np.zeros(17)

    for t in tags.split(' '):

        targets[label_map[t]] = 1 

    x_test0.append(cv2.resize(img, (input_dim, input_dim)))

    

x_test0 = np.array(x_test0, np.float16) / 255.
split = 192

# split = 35000

x_train, x_valid, y_train, y_valid = x_train0[:split], x_train0[split:], y_train0[:split], y_train0[split:]



# create the base pre-trained model

base_model = InceptionV3(weights=None, include_top=False, input_shape=(input_dim,input_dim,3))

# no weight initialization because Kaggle kernel is isolated from the internet, cannot download

# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(input_dim,input_dim))



# add a new top layer

x = base_model.output

x = Flatten()(x)

predictions = Dense(17, activation='sigmoid')(x)
# let's visualize layer names and layer indices to see how many layers 

# we should freeze

for i, layer in enumerate(base_model.layers):

    print(i, layer.name)
# this is the model we will train

model = Model(inputs=base_model.input, outputs=predictions)



# we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 172 layers and unfreeze the rest:

for layer in model.layers[:172]:

    layer.trainable = False

for layer in model.layers[172:]:

    layer.trainable = True
# Read in pre-trained weights - fast

# model.load_weights(obj_save_path + "weights_incv3.best.hdf5")
# we need to recompile the model for these modifications to take effect

# we use SGD with a low learning rate

from keras.optimizers import SGD

model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.

              optimizer=SGD(lr=0.01, momentum=0.9))
# Incorporate Callback features

# Checkpointing 

filepath= "weights_incv3.best.hdf5"

# filepath= obj_save_path + "weights_incv3.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)



# Early Stopping

earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=5, verbose=0, mode='auto') 



callbacks_list = [checkpoint, earlystop]
# the explicit split approach was taken so as to allow for a local validation

# model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)



# Fit the model (Add history so that the history may be saved)

history = model.fit(x_train, y_train,

          batch_size=32,

          epochs=1,

          verbose=1,

          callbacks=callbacks_list,

          validation_data=(x_valid, y_valid))



from sklearn.metrics import fbeta_score



p_train = model.predict(x_train0, batch_size=32,verbose=2)

p_test = model.predict(x_test0, batch_size=32,verbose=2)
def f2_score(y_true, y_pred):

    y_true, y_pred, = np.array(y_true), np.array(y_pred)

    return fbeta_score(y_true, y_pred, beta=2, average='samples')



def find_f2score_threshold(p_valid, y_valid, try_all=False, verbose=False):

    best = 0

    best_score = -1

    totry = np.arange(0.1,0.4,0.025) if try_all is False else np.unique(p_valid)

    for t in totry:

        score = f2_score(y_valid, p_valid > t)

        if score > best_score:

            best_score = score

            best = t

    if verbose is True: 

        print('Best score: ', round(best_score, 5), ' @ threshold =', best)

    return best



print(fbeta_score(y_train0, np.array(p_train) > 0.2, beta=2, average='samples'))

best_threshold = find_f2score_threshold(p_train, y_train0, try_all=True, verbose=True)
# Saving predicted probability and ground truth for Train Dataset

# Compute the best threshold externally

print(labels)

chk_output = pd.DataFrame()

for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:

    chk_output['class %d' % index] = p_train[:,index-1]

chk_output.to_csv('predicted_probability.csv', index=False)

for index in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:

    chk_output['class %d' % index] = y_train0[:,index-1]

chk_output.to_csv('true_label.csv', index=False)
values_test = (p_test > .222222)*1.0        # before multiplying by 1.0, this appears as an array of True and False

values_test = np.array(values_test, np.uint8)



print(values_test)

# Build Submission, using label outputted from long time ago

test_labels = []

for row in range(values_test.shape[0]):

    test_labels.append(' '.join(labels[values_test[row,:]==1]))

Submission_PDFModel = df_test.copy()

Submission_PDFModel.drop('tags', axis = 1)

Submission_PDFModel['tags'] = test_labels

Submission_PDFModel.to_csv('sub_pretrained_inception_v3_online.csv', index = False)