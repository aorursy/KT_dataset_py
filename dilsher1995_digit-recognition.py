# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

dataset_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
print(dataset_train.shape)

print(dataset_test.shape)
dataset_train.columns
X_train = dataset_train.iloc[:,1:]

Y_train = dataset_train.iloc[:,0]
Y_train.value_counts()
X_train.isna().any().sum()

dataset_test.isna().any().sum()
X_train.shape
X_train = X_train.values.reshape(-1,28,28,1)
X_train.shape
dataset_test = dataset_test.values.reshape(-1, 28, 28,1)
def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.values.reshape(-1)]

    return Y

Y_oh_train = convert_to_one_hot(Y_train, C = 10)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_oh_train, test_size = 0.1, random_state=42)
plt.imshow(X_train[1][:,:,0])
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))



model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))



model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))



model.add(Dense(84, activation = "relu"))

model.add(Dropout(0.2))

model.add(Dense(10, activation = "softmax"))
model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
result = model.fit(X_train, Y_train, batch_size = 132, epochs = 25, validation_data = (X_test, Y_test), verbose = 2)
result.history

pd.DataFrame.from_dict(result.history).plot()
test_img = plt.imshow(dataset_test[10][:,:,0])
dataset_test[0][:,:,0]

results = model.predict(dataset_test)
results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
results.iloc[10]
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("Digit_recogniser.csv",index=False)