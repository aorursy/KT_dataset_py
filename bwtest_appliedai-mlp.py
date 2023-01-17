# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas as pd

import numpy as np



from __future__ import absolute_import

from __future__ import print_function

from keras.utils import np_utils # For y values

# For plotting

%matplotlib inline

import seaborn as sns

# For Keras

from keras.models import Sequential

from keras.layers.core import Dense, Dropout

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler
train = pd.read_csv("../input/train_mnist.csv")

test = pd.read_csv("../input/test_mnist.csv")
train[1:10]
#Remove the first column from the data, as it is the label and put the rest in X

X = train.iloc[:, 1:].values

#Remove everything except the first column from the data, as it is the label and put it in y

y = train.iloc[:, :1].values
# The data does not load as a 28X28 image so there is no way to view it without reshaping it

X[0].shape
x_train = X.reshape(-1,28, 28)
x_train[1:2]
import matplotlib.pyplot as plt

#plot the first image in the dataset

plt.imshow(x_train[102])
#set input to the shape of one X value

dimof_input = X.shape[1]



# Set y categorical

dimof_output = int(np.max(y)+1)

y = np_utils.to_categorical(y, dimof_output)

x_train = X.reshape(-1,28, 28,1)
mlp_model = Sequential()

mlp_model.add(Dense(96, input_dim=dimof_input, kernel_initializer='uniform', activation='relu'))

mlp_model.add(Dense(256,kernel_initializer='uniform', activation='relu'))

mlp_model.add(Dense(96,kernel_initializer='uniform', activation='relu'))

mlp_model.add(Dense(dimof_output, kernel_initializer='uniform', activation='softmax'))

mlp_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
mlp_model.fit(

    X, y,

    validation_split=0.2,

    batch_size=10000, epochs=1, verbose=1)
mlp_model.summary()
test[1:5]
#use the model to predict the degits of the test data for mlp (784,1)

data = mlp_model.predict_classes(test, verbose=1)
data[1:10]
#creates a list of numbers from 0 to 27999 for the ImageID field

list_of_num = [i for i in range(0,28000)]



#Creates a dataframe that can be saved as a csv for submission

submission_data = pd.DataFrame(

    {'ImageId': list_of_num,

     'Label': data

    })
submission_data.to_csv('output.csv', sep=',',index=False)