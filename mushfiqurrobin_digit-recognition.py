# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_train.head()
df_test = pd.read_csv('../input/digit-recognizer/test.csv')

df_test.head()
import numpy as np

import tensorflow as tf

import pandas as pd

import matplotlib.pyplot as plt

import keras

from keras.models import Sequential

from keras.optimizers import RMSprop

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator
df_train.columns
df_test.columns
df_train.shape
df_test.shape
y_train = df_train["label"].astype('float32')

x_train = df_train.drop(labels = ["label"],axis = 1).astype('float32')
y_train.value_counts()
y_train.head()
x_train.head()
x_train.shape, y_train.shape
x_train.isnull().any().describe()
df_test.isnull().any().describe()
x_train = x_train/ 255.0

df_test = df_test/ 255.0
x_train = np.array(x_train).reshape(-1,28,28,1)

df_test = np.array(df_test).reshape(-1,28,28,1)
x_train.shape
df_test.shape
plt.imshow(x_train[4][:,:,0])
print(y_train[4])
y_train = keras.utils.to_categorical(y_train,10)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

#model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

#model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
model.summary()
datagen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False, 

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False,  

        zca_whitening=False, 

        rotation_range=10, 

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1, 

        horizontal_flip=False,  

        vertical_flip=False)



datagen.fit(x_train)
model.fit_generator(datagen.flow(x_train, y_train, batch_size = 25))
predict = model.predict(df_test)

final_pred = np.argmax(predict,axis = 1) 
final_pred[:10]
sample_sub = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

sample_sub.head()
sample_sub = pd.DataFrame({"ImageId": list(range(1,len(final_pred)+1)),

                         "Label": final_pred})
sample_sub[:10]
sample_sub.to_csv('final_submission.csv', index=False)