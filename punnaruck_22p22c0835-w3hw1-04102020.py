# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import json

import math

import os



import cv2

from PIL import Image

import numpy as np

from keras import layers

from keras.applications import DenseNet121, DenseNet201

from keras.callbacks import Callback, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.optimizers import Adam

from keras.models import Model

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, confusion_matrix, classification_report

from sklearn.utils import class_weight, shuffle

from sklearn.model_selection import KFold

import scipy

import tensorflow as tf

from tqdm import tqdm

from keras import backend as K

from keras.models import Model, save_model,load_model

import tensorflow as tf
pdDataset = pd.read_csv('/kaggle/input/super-ai-image-classification/train/train/train.csv')
def preprocess_image(image_path, IMG_SIZE):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))



    return image
pdDataset['id_path'] = pdDataset['id'].apply(lambda x : f'/kaggle/input/super-ai-image-classification/train/train/images/{x}')

pdDataset
IMG_SIZE = 300

BATCH_SIZE = 32

SEED = 33
N = pdDataset.shape[0]

x_train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)



for i, image_path in enumerate(tqdm(pdDataset['id_path'])):

#     print(image_path)

    x_train[i, :, :, :] = preprocess_image(image_path, IMG_SIZE)
y_train = pd.get_dummies(pdDataset['category']).values

y_train
def build_model_functional():

    densenet = DenseNet201(

        weights='imagenet',

        include_top=False,

        input_shape=(IMG_SIZE,IMG_SIZE,3)

    )



    base_model = densenet

    GAP_layer = layers.GlobalAveragePooling2D()

    drop_layer = layers.Dropout(0.6)

    dense_layer = layers.Dense(2, activation='sigmoid', name='final_output')

    

    x = GAP_layer(base_model.layers[-1].output)

    x = drop_layer(x)

    final_output = dense_layer(x)

    model = Model(base_model.layers[0].input, final_output)

    

    return model
# Define per-fold score containers <-- these are new

acc_per_fold = []

loss_per_fold = []

kf = KFold(n_splits = 10)

fold_no = 1



modelOne = build_model_functional() # with pretrained weights, and layers we want

modelOne.compile(

    loss='binary_crossentropy',

    optimizer=Adam(lr=0.000005),

    metrics=['accuracy']

)

for train, test in kf.split(x_train, y_train):



    # Generate a print

    print('------------------------------------------------------------------------')

    print(f'Training for fold {fold_no} ...')



    # Fit data to model

    history = modelOne.fit(x_train[train], y_train[train],

              batch_size=BATCH_SIZE,

              epochs=10,

              validation_data=(x_train[test], y_train[test]),

              verbose=1)

    

    # Generate generalization metrics

    scores = modelOne.evaluate(x_train[test], y_train[test], verbose=0)

    print(f'Score for fold {fold_no}: {modelOne.metrics_names[0]} of {scores[0]}; {modelOne.metrics_names[1]} of {scores[1]*100}%')

    acc_per_fold.append(scores[1] * 100)

    loss_per_fold.append(scores[0])



    # Increase fold number

    fold_no = fold_no + 1
# == Provide average scores ==

print('------------------------------------------------------------------------')

print('Score per fold')

for i in range(0, len(acc_per_fold)):

    print('------------------------------------------------------------------------')

    print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')

print('------------------------------------------------------------------------')

print('Average scores for all folds:')

print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')

print(f'> Loss: {np.mean(loss_per_fold)}')

print('------------------------------------------------------------------------')
def getValSet(test_set_folder):

    list_test_data = []

    for filename in os.listdir(test_set_folder):

        fullpath = os.path.join(test_set_folder, filename)

        if os.path.isfile(fullpath):

            dic_test = {}

            dic_test['id_path'] = fullpath

            dic_test['id'] = filename

            list_test_data.append(dic_test)



    pdTestSet = pd.DataFrame(list_test_data)



    return pdTestSet
def prepareValSet(pdValidate):

    N = pdValidate.shape[0]

    x_val_set = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)



    for i, image_id in enumerate(tqdm(pdValidate['id_path'])):

        x_val_set[i, :, :, :] = preprocess_image(image_id, IMG_SIZE)

    

    return x_val_set
pdValidate = getValSet("/kaggle/input/super-ai-image-classification/val/val/images")
x_validate = prepareValSet(pdValidate)
predictions = modelOne.predict(x_validate).argmax(axis=-1)

predictions
pdValidate['category'] = predictions

pdValidate
pdValidate[['id', 'category']].to_csv("val.csv", index=False)