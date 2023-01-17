# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import os 

import glob as gb

import cv2

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

import tensorflow as tf

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



sns.set(style='white', context='notebook', palette='deep')

tf.__version__
train_data = pd.read_csv('../input/nicht-mnist/Nicht MNIST.csv' ,header=None, index_col=0)

train_data
test_data = pd.read_csv('../input/nicht-data/test.csv',header=None, index_col=0)

test_data
train_data[1] = pd.Categorical(train_data[1])

train_data[1] = train_data[1].cat.codes

train_data[1]
df_test = train_data.sample(frac=0.3, random_state=7)

df_train = train_data.drop(df_test.index)
y_train = df_train.iloc[:,0]

x_train = df_train.iloc[:,1:]

y_val = df_train.iloc[:,0]

x_val = df_train.iloc[:,1:]
x_train
y_train.value_counts()
g = sns.countplot(y_train)
train_data.isnull().any().sum(), test_data.isnull().any().sum()

len(y_train.value_counts())
x_train = x_train / 255.0 

x_val = x_val / 255.0 
x_train_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in x_train.iterrows()] ] )

x_val_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in x_val.iterrows()] ] )
model = Sequential()



model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
model.build()

model.summary()
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "SparseCategoricalCrossentropy", metrics=["accuracy"])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 60

batch_size = 100
datagen = ImageDataGenerator(

        rotation_range=9,

        zoom_range = 0.1,

        width_shift_range=0.1,

        height_shift_range=0.1)





datagen.fit(x_train_np)
history = model.fit_generator(datagen.flow(x_train_np,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val_np,y_val),

                              verbose = 2, steps_per_epoch=x_train_np.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
model.evaluate(x_val_np,  y_val, verbose=2)
test_data = test_data / 255.0

test_data
x_test_np = np.vstack([[np.array(r).astype('uint8').reshape(28,28, 1) for i, r in test_data.iterrows()] ] )
model.predict(x_test_np)
preds = np.argmax(model.predict(x_test_np), axis=1).tolist()
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

pred_labes = pd.Series([class_labels[p] for p in preds])

pred_labes
out_df = pd.DataFrame({

    'Id': test_data.index,

    'target': pred_labes

})

out_df
out_df.to_csv('my_submission.csv', index=False)