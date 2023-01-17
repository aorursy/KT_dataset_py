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
# Importing the necessary dependencies



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm, tqdm_notebook

import cv2

import os

from keras.models import Sequential

from keras.utils import to_categorical

from keras.layers import Dense, Dropout, Activation, Conv2D, Flatten

from keras.callbacks import ModelCheckpoint

train_dir = "../input/train/train/"

test_dir = "../input/test/test/"

train_df = pd.read_csv('../input/train.csv')
# Linking the images as per the csv file, and using the pixels as features.

# This part is inspired from another kernel 'Keras _Transfer_VGG16' of the same competetion



features = []

target = []

images = train_df['id'].values

for img_id in tqdm_notebook(images):

    features.append(cv2.imread(train_dir + img_id))    

    target.append(train_df[train_df['id'] == img_id]['has_cactus'].values[0])  

features = np.asarray(features)

features = features.astype('float32')

features /= 255

target = np.asarray(target)
# Specifying the model



model = Sequential()

model.add(Conv2D(15, kernel_size=3, activation='relu', input_shape=(32,32,3), padding='same'))

model.add(Conv2D(15, kernel_size=3, activation='relu', padding='same'))

model.add(Conv2D(15, kernel_size=3, activation='relu', padding='same'))

model.add(Flatten())

model.add(Dense(1, activation='sigmoid'))



# Model Summary

model.summary()
# Compiling the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



# Storing the optimal parameters

# This checkpoint object will store the model parameters in the file "weights.hdf5"

checkpoint = ModelCheckpoint('weights.hdf5', monitor='val_loss', save_best_only=True)

# Store in a list to be used during training

callbacks_list = [checkpoint]
# Training the model

training = model.fit(features, target, validation_split=0.1, epochs=15, callbacks = callbacks_list)



# Visualizing the losses vs. epochs

plt.plot(training.history['loss'])

plt.plot(training.history['val_loss'])

plt.show()
# Using the pixels of the test images as features



test_features = []

Test_images = []

for img_id in tqdm_notebook(os.listdir(test_dir)):

    test_features.append(cv2.imread(test_dir + img_id))     

    Test_images.append(img_id)

test_features = np.asarray(test_features)

test_features = test_features.astype('float32')

test_features /= 255
# Running the model over the test images



test_predictions = model.predict(test_features)

submissions = pd.DataFrame(test_predictions, columns=['has_cactus'])

submissions['has_cactus'] = submissions['has_cactus'].apply(lambda x: 1 if x > 0.75 else 0)

submissions['id'] = ''

cols = submissions.columns.tolist()

cols = cols[-1:] + cols[:-1]

submissions=submissions[cols]

for i, img in enumerate(Test_images):

    submissions.set_value(i,'id',img)

# Saving the output file



submissions.to_csv('submission.csv',index=False)
