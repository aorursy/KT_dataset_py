# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_set = pd.read_csv("../input/train.csv")

test_set = pd.read_csv("../input/test.csv")
Y_train = train_set["label"]



# Drop 'label' column

X_train = train_set.drop(labels = ["label"],axis = 1) 



# free some space

del train_set
X_train.isnull().any().describe()
X_train=X_train/255.0
X_train = X_train.values.reshape(-1,28,28,1)

test_set = test_set.values.reshape(-1,28,28,1)




import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



np.random.seed(2)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop ,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



Y_train = to_categorical(Y_train, num_classes = 10)


X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
g = plt.imshow(X_train[0][:,:,0])
def getmodel():

    model=Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', strides=2

                     ,activation ='relu', input_shape = (28,28,1)))

    model.add(Conv2D(filters = 48, kernel_size = (4,4),padding = 'valid', strides=2

                     ,activation ='relu'))

    model.add(Dropout(0.25))





    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', strides=1

                     ,activation ='relu'))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(576, activation = "relu"))

    model.add(Dropout(0.25))

    model.add(Dense(100,activation='relu'))

    

    model.add(Dense(10, activation = "softmax"))

    return model
model=getmodel()

model.compile(optimizer=Adam(epsilon=1e-08,lr=0.0005), loss='categorical_crossentropy',metrics=["accuracy"])
history = model.fit(X_train, Y_train, batch_size = 64, epochs = 30, 

         validation_data = (X_val, Y_val), verbose = 1)
results = model.predict(test_set)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)
print (results)