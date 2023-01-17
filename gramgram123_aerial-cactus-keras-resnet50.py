# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



input_dir = os.path.dirname("../input/aerial-cactus-identification/")

print(os.listdir(input_dir))



train = pd.read_csv(os.path.join(input_dir, "train.csv"))

test = pd.read_csv(os.path.join(input_dir, "sample_submission.csv"))

                                

# Any results you write to the current directory are saved as output.
images = np.array(train.id.values)

labels = np.array(train.has_cactus)



# split data into train/val - train-12000, val-5499

train_images = images[:12000]

train_labels = labels[:12000]



val_images = images[12001:]

val_labels = labels[12001:]
import cv2

img_base_dir = "train/train/"

trn_data=[]

for name in train_images:

    img_path = img_base_dir + name

    i = cv2.imread(os.path.join(input_dir, img_path))  

    trn_data.append(i)



trn_data = np.array(trn_data)

print('trn data shape', trn_data.shape)



val_data=[]

for name in val_images:

    img_path = img_base_dir + name

    i = cv2.imread(os.path.join(input_dir, img_path))  

    val_data.append(i)



val_data = np.array(val_data)

print('val data shape', val_data.shape)



test_data=[]

test_img_base_dir = "test/test/"

test_images =  np.array(test.id.values)

test_labels = np.array(test.has_cactus)



for name in test_images:

    img_path = test_img_base_dir + name

    i = cv2.imread(os.path.join(input_dir, img_path))  

    test_data.append(i)

    

test_data = np.array(test_data)

print('test data shape', test_data.shape)
print(test_images)
import random

import numpy as np

from keras.applications.resnet50 import preprocess_input



def generator(features, labels, batch_size):

    # Create empty arrays to contain batch of features and labels#

    batch_features = np.zeros((batch_size, 32, 32, 3))

    batch_labels = np.zeros((batch_size,1))

    num_features = len(features)

    

#     print('number of features: ', num_features)

    

    while True:

        for i in range(batch_size):

            index= random.choice(np.arange(num_features))

            _data = keras.applications.resnet50.preprocess_input(features[index])



            batch_features[i] = _data

            batch_labels[i] = labels[index]

            

#             print(batch_features)

#             print(batch_labels)

            

        yield batch_features, batch_labels

# trn_gen = generator(trn_data, train_labels, 32)

# print(trn_data.shape[1:])
import keras

from pathlib import Path

from keras.models import Sequential, Model

from keras.layers import Dense, Flatten, GlobalAveragePooling2D



opt = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5)



weights = Path('../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

base_model = keras.applications.resnet50.ResNet50(include_top=False, weights=weights, input_shape=trn_data.shape[1:])

x = base_model.output

# model.add(Flatten())

# model = GlobalAveragePooling2D()(model.output)

# model.add(Dense(units = 1, activation='sigmoid'))

x = GlobalAveragePooling2D()(x)

x = Dense(1024, activation='relu')(x)

predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs = base_model.input, outputs = predictions)



model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
n_epochs = 10

batch_size = 64



n_steps = int(trn_data.shape[0]/batch_size)

n_val = int(val_data.shape[0]/batch_size)



trn_gen = generator(trn_data, train_labels, batch_size)

val_gen = generator(val_data, val_labels, batch_size)



fit_history = model.fit_generator(generator=trn_gen, validation_data=val_gen, verbose=1, epochs=n_epochs, steps_per_epoch=n_steps, validation_steps=n_val, initial_epoch=0)

# acc_val = history.history['val_acc']
results = []

for i in test_data:

    pred_img = i[np.newaxis, ...] # add batch dimension

    pred_img = keras.applications.resnet50.preprocess_input(pred_img)

    res = model.predict(pred_img, batch_size=1)

    results.append(res[0][0])



submission = {'id': test_images, 'has_cactus':results} 

submission = pd.DataFrame(submission)

print(submission.head())
submission.to_csv("submission.csv", index=False)