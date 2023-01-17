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
from tensorflow import keras

from tensorflow.keras import utils as np_utils

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

from tensorflow.keras.models import Sequential 



from tensorflow.keras.layers import Convolution2D



from tensorflow.keras.layers import MaxPooling2D



from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Activation, Dropout, Dense 

from tensorflow.keras import backend as K 

from sklearn.metrics import accuracy_score  

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

import numpy as np

x = ('/kaggle/input/intel-image-classification/seg_train/seg_train')

y = ('/kaggle/input/intel-image-classification/seg_test/seg_test')

z = ('/kaggle/input/intel-image-classification/seg_pred/seg_pred')

train_batches = ImageDataGenerator().flow_from_directory(x,target_size=(150,150),

                                                         classes=['glacier','sea','buildings','forest','street','mountain'],

                                                         batch_size=150)



test_batches =  ImageDataGenerator().flow_from_directory(y,target_size=(150,150),

                                                         classes=['glacier','sea','buildings','forest','street','mountain'],

                                                         batch_size=528)

#model = Sequential()

#model.add(Convolution2D(filters = 32, kernel_size = (3, 3),

#input_shape = (150, 150, 3)))

#model.add(Flatten())

#model.add(Dense(units = 6, activation = 'softmax'))



#model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    #epochs=2)





model = Sequential()

model.add(Convolution2D(filters = 64, kernel_size = (3, 3),

input_shape = (150, 150, 3)))

model.add(Convolution2D(filters = 64,kernel_size = (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filters = 128,kernel_size = (3, 3), activation='relu'))

model.add(Convolution2D(filters = 128,kernel_size = (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filters = 256,kernel_size = (3, 3), activation='relu'))

model.add(Convolution2D(filters = 256,kernel_size = (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Convolution2D(filters = 512,kernel_size = (3, 3), activation='relu'))

model.add(Convolution2D(filters = 512,kernel_size = (3, 3), activation='relu'))

model.add(Convolution2D(filters = 512,kernel_size = (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))



model.add(Flatten())

model.add(Dense(units = 6, activation = 'softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.fit_generator(train_batches,steps_per_epoch =12,validation_data = test_batches,validation_steps=11,

                    epochs=20)
model.save('sign_model.h5')
predictions=model.predict_generator(test_batches,steps=1,verbose=1)