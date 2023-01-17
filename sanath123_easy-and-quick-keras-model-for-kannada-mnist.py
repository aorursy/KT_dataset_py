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
import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib.image as mpimg

import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import ReduceLROnPlateau



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
train_df = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test_df = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')

pred_df = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')
X_train = train_df.drop(['label'], axis = 1)

Y_train = train_df['label']

X_pred = pred_df.drop(['id'], axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
X_train = X_train / 255.0

Y_train = Y_train / 255.0

X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
Y_train = Y_train.to_numpy()
model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(16, (3,3), activation='relu'),

    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')

])

model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])

model.summary()
test=pd.read_csv('../input/Kannada-MNIST/test.csv')
test_id=test.id



test=test.drop('id',axis=1)

test=test/255

test=test.values.reshape(-1,28,28,1)
y_pre=model.predict(test)     

y_pre=np.argmax(y_pre,axis=1) 
sample_sub=pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
sample_sub['label']=y_pre

sample_sub.to_csv('submission.csv',index=False)