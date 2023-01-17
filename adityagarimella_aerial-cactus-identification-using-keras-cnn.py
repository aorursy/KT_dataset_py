# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import numpy as np

import pandas as pd 

from PIL import Image

from glob import glob

import matplotlib.pyplot as plt

from keras.optimizers import Adam

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, Activation



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
base_dir = "../input"
train_df = pd.read_csv(base_dir + '/train.csv')
train_df.dtypes
train_df.head()
os.listdir(base_dir)
train_images_dir = base_dir + '/train/train'

test_images_dir = base_dir + '/test/test'
train_images_dir
len(os.listdir(train_images_dir))
os.listdir(train_images_dir)[0]
images_paths_list = glob(train_images_dir + '/*.jpg')
len(images_paths_list)
images_paths_list[0]
image_id_path_dict = {os.path.basename(x): x for x in images_paths_list}
list(image_id_path_dict.keys())[:3]
exact_paths_list = [image_id_path_dict[x] for x in list(train_df["id"])]
len(exact_paths_list)
exact_paths_list[0]
train_df["paths"] = exact_paths_list
train_df.head(1).values
#Get and store images by resizing into 32 x 32

train_df['images'] = train_df['paths'].map(lambda x: np.array(Image.open(x).resize((32,32))))
len(train_df)
train_df.head(1).values
#Normalize the images

train_df['images'] = train_df['images'] / 255
train_df["images"][0].shape
train_df["has_cactus"].value_counts()
type(train_df["has_cactus"].value_counts())
# Creating the test df in similar way

test_df = pd.read_csv(base_dir + '/sample_submission.csv')
test_df.head()
test_images_paths_list = glob(test_images_dir + '/*.jpg')
len(test_images_paths_list)
test_image_id_path_dict = {os.path.basename(x): x for x in test_images_paths_list}
test_exact_paths_list = [test_image_id_path_dict[x] for x in list(test_df["id"])]
test_df["paths"] = test_exact_paths_list
test_df.head()
test_df['images'] = test_df['paths'].map(lambda x: np.array(Image.open(x)))
test_df['images'] = test_df['images'] / 255
test_df["images"][0].shape
#Start modelling



x = train_df['images']

y = train_df['has_cactus']
type(x)
x = np.array(list(x))

y = np.array(list(y))
x.shape,y.shape
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.15, random_state=1234)
# Use to reduce learning rate 

lr = ReduceLROnPlateau(monitor= 'val_acc',

                              patience=3,

                              verbose=1,

                              factor=0.5)
callbacks = [lr]
input_shape = x_train.shape[1:]



model = Sequential([Conv2D(32, kernel_size=(2,2), activation='relu', padding='same', input_shape = input_shape),

                  Conv2D(32, kernel_size=(2,2), activation='relu', padding='same'),

                  MaxPool2D(pool_size=(2,2)),

                  Dropout(0.2),

                  

                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),

                  Conv2D(64, kernel_size=(2,2),activation='relu', padding='same'),

                  MaxPool2D(pool_size=(2,2)),

                  Dropout(0.2),

                  

                  Conv2D(128, kernel_size=(2,2), activation='relu', padding='same'),

                  Conv2D(256, kernel_size=(2,2), activation='relu', padding='same'),

                  MaxPool2D(pool_size=(2,2)),

                  Dropout(0.5),

                  

                  Flatten(),

                  Dense(64, activation='relu'),

                  Dropout(0.5),

                  Dense(28, activation='relu'),

                  Dropout(0.5),

                  Dense(1, activation='sigmoid')

                 ])
model.compile(optimizer=Adam(lr=0.001), loss = "binary_crossentropy", metrics=['acc'])
#fit the cnn model

model_stats = model.fit(x_train, y_train, epochs=30, batch_size=64, validation_data=(x_valid,y_valid), callbacks=callbacks)
test_pred = model.predict_classes(np.array(list(test_df['images'])))
type(test_pred)
test_pred.shape
test_df['has_cactus'] = np.squeeze(test_pred)
test_df.head()
test_df.head(1).values
submission_df = test_df[['id','has_cactus']]
submission_df.to_csv('cactus_detections.csv', index=False)