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

import matplotlib.pyplot as plt

from PIL import Image



from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, LeakyReLU, Flatten, BatchNormalization

from keras.optimizers import Adam



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test  = pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

train.shape, test.shape
y = train.pop('label')

test = test.drop('id',axis=1)
train = train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
train = train/255

test = test/255

train.shape, test.shape
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size = 0.2)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.05)
# model

from functools import partial



DefConv2D = partial(Conv2D, kernel_size=(3,3), padding='same')



model = Sequential([

    Conv2D(32,input_shape = (28,28,1), kernel_size=(3,3)),

    LeakyReLU(alpha=0.1),

    BatchNormalization(),

    

    DefConv2D(64),

    DefConv2D(64),

    LeakyReLU(alpha=0.1),

    BatchNormalization(),

    MaxPool2D(2,2),

    

    DefConv2D(128),

    DefConv2D(128),

    LeakyReLU(alpha=0.1),

    BatchNormalization(),

    MaxPool2D(2,2),

    

    DefConv2D(256),

    DefConv2D(256),

    LeakyReLU(alpha=0.1),

    BatchNormalization(),

    MaxPool2D(2,2),

    

    Flatten(),

    Dense(256, activation='relu'),

    Dropout(0.5),

    Dense(10,activation='softmax')

])
model.summary()
optimizer = Adam(learning_rate=0.001)

model.compile(loss='sparse_categorical_crossentropy',

              optimizer=optimizer,

              metrics=['accuracy'])
history = model.fit(x_train,y_train, validation_data = (x_val,y_val) , epochs=15, batch_size=1024)
pd.DataFrame(history.history).plot(figsize=(8,5))

plt.grid(True)

plt.gca().set_ylim(0,1)

plt.show()
y_pred = model.predict(x_test)

y_pred = np.argmax(y_pred,axis=1)
conf_mat = confusion_matrix(y_test.values,y_pred)

conf_mat = pd.DataFrame(conf_mat,index=range(0,10),columns=range(0,10))
conf_mat
# submission

y_predict = model.predict(test) 

y_predict = np.argmax(y_predict,axis=1)



sample_sub = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

sample_sub['label'] = y_predict

sample_sub.to_csv('submission.csv',index=False)