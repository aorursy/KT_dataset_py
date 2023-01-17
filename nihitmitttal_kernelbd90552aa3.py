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
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd

import numpy as np

import csv

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D, Activation

from tensorflow.keras.callbacks import *

from tensorflow.keras import optimizers

from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding

from tensorflow.keras.optimizers import *

from sklearn.model_selection import train_test_split

dataset=pd.read_csv('drive/My Drive/train.csv')

testdataset=pd.read_csv('drive/My Drive/test.csv')

label = dataset["label"]

feature = dataset.drop(labels = ["label"],axis = 1)
label = to_categorical(label, num_classes = 10)

feature = feature / 255.0

testdataset = testdataset / 255.0

feature = feature.values.reshape(-1,28,28,1)

feature_train, feature_val, label_train, label_val = train_test_split(feature, label, test_size = 0.1, stratify=label)

from keras.layers import LeakyReLU





model = Sequential()



#1. LAYER

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', input_shape=(28, 28, 1)))

model.add(BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



#2. LAYER

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(MaxPool2D(pool_size=(2, 2)))



#3. LAYER

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



#4. LAYER

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same'))

model.add(BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))



model.add(MaxPool2D(pool_size=(2, 2)))



#FULLY CONNECTED LAYER

model.add(Flatten())

model.add(Dense(256))

model.add(BatchNormalization())

model.add(keras.layers.LeakyReLU(alpha=0.3))

model.add(Dropout(0.25))



#OUTPUT LAYER

model.add(Dense(10, activation='softmax'))



# Define the optimizer and compile the model

#optimizer = RMSprop(lr = 0.001, rho=0.9, epsilon=1e-08, decay=0.0)

optimizer = Adam(lr = 0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,amsgrad=False)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss',verbose=1,

                                            patience=3, 

                                            factor=0.5, 

                                            min_lr=0.00001

                                            )

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1,

                               save_best_only=True) 



print (model.summary())
batch_size = 100

history = model.fit(feature_train,label_train, batch_size=batch_size,

                     shuffle=True,

                     epochs=50, validation_data = (feature_val,label_val),steps_per_epoch=feature_train.shape[0] // batch_size,

                     verbose = 2, callbacks= [learning_rate_reduction, checkpointer])
model.load_weights('model.weights.best.hdf5')

testdataset=testdataset.drop(labels = ["id"],axis = 1)
testdataset = testdataset.values.reshape(-1,28,28,1)

results = model.predict(testdataset)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="pred")

submission = pd.concat([pd.Series(range(1,10001),name = "id"),results],axis = 1)

submission.to_csv("submission_final.csv",index=False)

from google.colab import files

files.download('submission_final.csv')