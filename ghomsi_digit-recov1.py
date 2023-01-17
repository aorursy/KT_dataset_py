# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#load packages

import keras 

import matplotlib.pyplot as plt

import seaborn as sns

from keras.models import Sequential

from keras.layers import Dense, Activation ,Conv2D , Dropout , Flatten , MaxPooling2D

from skimage.transform import rotate

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
import pandas as pd

#load data

sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

data_test = pd.read_csv("../input/digit-recognizer/test.csv")

data_train = pd.read_csv("../input/digit-recognizer/train.csv")
label_train = data_train['label']

data_train = data_train.drop('label' , axis=1)
#trasform to numpy type

data_test = data_test.to_numpy()

data_train = data_train.to_numpy()

label_train = label_train.to_numpy()
data_train.shape
#encode label to onehot

encoder  = OneHotEncoder()

label_train = label_train.reshape(-1,1)

oneHotLabel = encoder.fit_transform(label_train).toarray()



#create a train data and validation data

X_train ,val_test , y_train , y_test = train_test_split(data_train , oneHotLabel)



#reshape to (,28,28,1)

X_train = X_train.reshape(X_train.shape[0],28,28,1)

val_test = val_test.reshape(val_test.shape[0],28,28,1)



#scaling data

X_train = X_train / 255

val_test = val_test / 255

#conv layers in keras

batch_size = 100

n_epochs = 100



model = Sequential()



model.add(Conv2D(32,(2,2),strides=1,padding='same' ,

                 input_shape=(28,28,1)))

model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same', data_format=None))

model.add(Dropout(0.25))



model.add(Conv2D(32,(3,3),strides=1,padding='same'))

model.add(MaxPooling2D(pool_size=(2,2) , strides=1))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(500))

model.add(Activation('relu'))

model.add(Dropout(0.5))





model.add(Dense(10))

model.add(Activation('softmax'))



opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss='categorical_crossentropy' , optimizer=opt , metrics=['accuracy'])
model.fit(X_train,y_train, batch_size=batch_size, epochs=n_epochs, validation_data=(val_test , y_test))
data_test = data_test.reshape(data_test.shape[0],28,28,1)
predictions = model.predict(data_test)
predictions = encoder.inverse_transform(predictions)
predictions = np.int64(predictions)
predictions.shape
sample_submission.head()

num = np.arange(28000)

preds = np.c_[num,predictions]

preds  = pd.DataFrame(preds , columns=['ImageId','Label'] , index=None)

preds
preds.to_csv('samples.csv' , index=False)