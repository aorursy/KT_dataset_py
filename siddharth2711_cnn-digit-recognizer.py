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
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from sklearn.model_selection import train_test_split

import pandas as pd

import numpy as np
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.info()
y_train = train['label']

X_train = train.drop(labels = 'label',axis = 1)
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train = X_train / 255.0

test = test / 255.0
num_classes = 10

y_train = to_categorical(y_train,num_classes)
seed = 43

X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=seed)
classifier = Sequential()

classifier.add(Conv2D(32,(3,3),padding = 'Same',activation = 'relu',input_shape = (28,28,1)))

classifier.add(Conv2D(32,(3,3),padding = 'Same',activation = 'relu'))

classifier.add(MaxPool2D(pool_size = (2,2)))

classifier.add(Dropout(0.25))



classifier.add(Conv2D(64,(3,3),padding = 'Same',activation = 'relu'))

classifier.add(Conv2D(64,(3,3),padding = 'Same',activation = 'relu'))

classifier.add(MaxPool2D(pool_size = (2,2)))

classifier.add(Dropout(0.25))



classifier.add(Flatten())

classifier.add(Dense(256,activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(10,activation = 'softmax'))
classifier.compile(optimizer = RMSprop(lr = 0.001),loss = 'categorical_crossentropy',metrics = ['accuracy'])
epochs = 30

batch_size = 86

IDG = ImageDataGenerator(

        rotation_range=10,

        shear_range=0.3,

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1,  

        horizontal_flip=False,  

        vertical_flip=False) 





IDG.fit(X_train)
history = classifier.fit_generator(IDG.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size)
results = classifier.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_classifier.csv",index=False)