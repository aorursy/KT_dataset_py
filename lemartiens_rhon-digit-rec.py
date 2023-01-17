# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop, Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

path_df = '/kaggle/input/digit-recognizer/'
train_df= pd.read_csv(path_df + 'train.csv')

test_df = pd.read_csv(path_df + 'test.csv')
y = train_df.label

X = train_df.drop(columns='label')
# Rescaling

X = X / 255.0

test_df = test_df / 255.0

# Reshaping

X = X.values.reshape(-1,28,28,1)#Reshaping

test_df = test_df.values.reshape(-1,28,28,1)



y = to_categorical(y, num_classes = 10)#Encoding
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.25, random_state=1337)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

          

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
optimizer = Adam()

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
gogo = model.fit(X_train, Y_train, batch_size = 64, epochs = 5, 

                 validation_data = (X_val, Y_val), verbose = 1)
sample_submissions = pd.read_csv(path_df + 'sample_submission.csv')

sub = model.predict(test_df)

sub = np.argmax(sub,axis = 1)

sample_submissions['Label'] = sub

sample_submissions.to_csv('./submission.csv', index=False)