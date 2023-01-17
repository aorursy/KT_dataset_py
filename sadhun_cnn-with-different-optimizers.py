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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.head(2)
test.head(2)
train = np.array(train)
test = np.array(test)

train_x = train[:, 1:]
train_y = train[:, 0]

# Normalize the data
train_x = train_x / 255.0
test = test / 255.0


train_x
#Reshape
train_x = train_x.reshape(-1,28,28,1)
test = test.reshape(-1,28,28,1)
train_y
train_y = to_categorical(train_y)
train_y
print(train_x.shape)
g = plt.imshow(train_x[1][:,:,0])
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
#Trying RMSProp optimizer

optimizer_rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model
model.compile(optimizer = optimizer_rms , loss = "categorical_crossentropy", metrics=["accuracy"])
BATCH_SIZE=64
EPOCH=5
history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.2, shuffle=True)
from tensorflow.keras.optimizers import Adam
optimizer_adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,name='Adam')
# Compile the model
model.compile(optimizer = optimizer_adam , loss = "categorical_crossentropy", metrics=["accuracy"])
# Since target variable has multiple classes, loss fnctn can be categorical cross entropy only
history_adam = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.2, shuffle=True)
#Stochastic gradient with momentum and Nesterov
#we can slo use momentum optimzer (sgd+momentum)
from tensorflow.keras.optimizers import SGD
optimizer_sgd=SGD(learning_rate=0.01, momentum=0.9, nesterov=True, name='SGD')
model.compile(optimizer = optimizer_sgd , loss = "categorical_crossentropy", metrics=["accuracy"])
history_sgd = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.2, shuffle=True)
    