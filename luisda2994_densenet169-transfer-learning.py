# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import cv2

import pandas as pd

import numpy as np

import json



from tqdm import tqdm, tqdm_notebook



import keras

from keras.models import Sequential, Model

from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

from keras.preprocessing.image import ImageDataGenerator



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import os




#print(os.listdir("../input/densenet-keras/DenseNet-BC-169-32-no-top.h5"))

train_dir = "../input/aerial-cactus-identification/train/train/"

test_dir = "../input/aerial-cactus-identification/test/test/"

train_df = pd.read_csv('../input/aerial-cactus-identification/train.csv')
X = []

y = []

imges = train_df['id'].values

for img_id in tqdm_notebook(imges):

    X.append(cv2.imread(train_dir + img_id))    

    y.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  

X = np.asarray(X)

X = X.astype('float32')

X /= 255

#y = np.asarray(y)

print('Shape of X tensor: ',X.shape)

print('Length of target list: ',len(y))



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1)

y_train = keras.utils.to_categorical(y_train,2)

y_val = keras.utils.to_categorical(y_val,2)



datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)



val_datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)

datagen.fit(X_train)

val_datagen.fit(X_val)
base_model = keras.applications.densenet.DenseNet169(include_top=False, input_shape=(32,32,3), 

                                                     weights = '../input/densenet-keras/DenseNet-BC-169-32-no-top.h5')

base_model.trainable = False
#To add the final classifier

add_model = Sequential()

add_model.add(Flatten(input_shape=base_model.output_shape[1:]))

add_model.add(Dense(256, activation='relu'))

add_model.add(Dropout(0.2))

add_model.add(Dense(2, activation='softmax'))



model = Model(inputs = base_model.input, outputs = add_model(base_model.output))

model.compile(optimizer='adam',loss='categorical_crossentropy', 

                metrics=['accuracy'])
# fits the model on batches with real-time data augmentation:

model.fit_generator(datagen.flow(X_train, y_train, batch_size=16),

                    steps_per_epoch=len(X_train) / 16, epochs=100,

                    validation_data = val_datagen.flow(X_val,y_val), validation_steps=len(X_val) / 16, verbose = 1)
X_tst = []

Test_imgs = []

for img_id in tqdm_notebook(os.listdir(test_dir)):

    X_tst.append(cv2.imread(test_dir + img_id))     

    Test_imgs.append(img_id)

X_tst = np.asarray(X_tst)

X_tst = X_tst.astype('float32')

X_tst /= 255



# Prediction

test_predictions = model.predict(X_tst)



#print(test_predictions)

test_predictions_2 = np.argmax(test_predictions, axis = 1)

print(test_predictions_2)

print(test_predictions_2.shape)
sub_df = pd.DataFrame(test_predictions_2, columns=['has_cactus'])

sub_df['has_cactus'] = sub_df['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)



sub_df['id'] = ''

cols = sub_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

sub_df=sub_df[cols]



for i, img in enumerate(Test_imgs):

    sub_df.set_value(i,'id',img)

    
sub_df.head()




sub_df.to_csv('submission.csv',index=False)


