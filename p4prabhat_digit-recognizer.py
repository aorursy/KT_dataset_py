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



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

np.random.seed(2)

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

sns.set(style="white", context="notebook", palette= "deep")
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)
del train
g = sns.countplot(Y_train) 

Y_train.value_counts()
# somewhat similar counts for all 10 digits
# looking for null values

X_train.isnull().sum()
test.isnull().sum()
# there is no missing value inside
X_train = X_train/ 255.0

test = test/ 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = to_categorical (Y_train,num_classes=10)

# labels are 10 digits from 0 to 9
random_seed = 2
X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train, test_size= 0.2, random_state= random_seed)
g = plt.imshow(X_train[0][:,:,0])
model = Sequential()

model.add(Conv2D(filters = 32 , kernel_size=(5,5), padding="Same", activation = "relu", input_shape= (28,28,1)))

model.add(Conv2D(filters = 32 , kernel_size=(5,5), padding="Same", activation = "relu"))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64 , kernel_size=(3,3), padding="Same", activation = "relu"))

model.add(Conv2D(filters = 64 , kernel_size=(3,3), padding="Same", activation = "relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10,activation="softmax"))

# define the optimizer

optimizer = RMSprop(lr=0.001, rho= 0.9, decay =0.0 )
#compile the model

model.compile(optimizer = optimizer, loss= "categorical_crossentropy", metrics=["accuracy"])
#learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor="val_acc",patience=3, verbose=1, factor=0.5,min_lr=0.00001)
epochs = 1

batch_size = 86
history = model.fit(X_train,Y_train,batch_size=batch_size, epochs=epochs, validation_data= (X_val,Y_val),verbose =2)
results = model.predict(test)

results = np.argmax(results,axis=1)

results= pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name ="ImageId"),results],axis=1)