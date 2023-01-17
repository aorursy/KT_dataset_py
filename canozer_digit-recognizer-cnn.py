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
#importing necessary libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as pltimg

import seaborn as sns

import missingno as msno

sns.set()
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



train
X_train = train.drop(['label'],axis = 1)

y_train = train['label']
test
X_test = test
#Checking for null values in train and test set



print('Number of null values in training set is : ',train.isnull().sum().unique())

print('Number of null values in test set is : ',test.isnull().sum().unique())
plt.figure(figsize = (12,8))

sns.distplot(y_train)
#reshaping

X_train_reshaped = X_train.values.reshape(-1,28,28,1)

X_test_reshaped = X_test.values.reshape(-1,28,28,1)
#Normalising

X_train_normalised = X_train_reshaped/255.

X_test_normalised = X_test_reshaped/255.
plt.figure(figsize = (15,15))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(X_train_normalised[i][:,:,0], cmap = 'gray')

    plt.xticks([])

    plt.yticks([])

    plt.title('Label {}'.format(y_train[i]))
from tensorflow.keras.utils import to_categorical
# one hot encoding

y_train_encoded = to_categorical(y_train, num_classes = 10)
#splitting the data

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train_normalised, y_train_encoded, test_size = 0.2, random_state = 42)
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D

from tensorflow.keras.layers import BatchNormalization,Flatten,Dropout

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
# CNN

model = Sequential([Conv2D(filters=32, kernel_size=(5,5), \

                           activation='relu', input_shape=(28,28,1)),

    MaxPool2D(pool_size=(2,2)),

    Conv2D(filters=64, kernel_size=(3,3), \

                           activation='relu', input_shape=(28,28,1)),

    MaxPool2D(pool_size=(2,2), strides=(2,2)),

    Flatten(),

    Dense(256, activation='relu'),

    Dense(10, activation='softmax')

])



model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)



model.summary()
callbacks = [ 

    EarlyStopping(monitor = 'loss', patience = 6), 

    ReduceLROnPlateau(monitor = 'loss', patience = 4)

]
model.fit(X_train,y_train,

         batch_size = 64,

         epochs = 100,

         verbose = 1,

         validation_data = (X_test,y_test),

         callbacks = callbacks)
#Metric on validation set

score = model.evaluate(X_test,y_test,verbose = 0)

print('The loss on validation set is {0} and the accuracy is {1}'.format(round(score[0],3),round(score[1],3)))
# predict results

results = model.predict(X_test_normalised)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name='Label')
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("submission.csv",index=False)