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

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns



np.random.seed(1)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.describe()
y_train = df_train['label']

x_train = df_train.drop(labels = ['label'] , axis=1)

del df_train
g = sns.countplot(y_train)
y_train.value_counts()
x_train = x_train/255.0

df_test = df_test/255.0
x_train.isnull().any().describe()
df_test.isnull().any().describe()
x_train = x_train.values.reshape(-1 , 28 , 28 ,1)

df_test = df_test.values.reshape(-1 , 28 ,28 ,1)
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train , num_classes = 10)
from sklearn.model_selection import train_test_split

x_train  , x_val , y_train , y_val = train_test_split(x_train , y_train , test_size = .1 , random_state = 0)
import matplotlib.image as mpimg

g = plt.imshow(x_train[0][: , : , 0])
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D



classifier = Sequential()



classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

classifier.add(MaxPool2D(pool_size=(2,2)))

classifier.add(Dropout(0.25))







classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

classifier.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

classifier.add(Dropout(0.25))







classifier.add(Flatten())

classifier.add(Dense(256, activation = "relu"))

classifier.add(Dropout(0.5))

classifier.add(Dense(10, activation = "softmax"))
classifier.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 30 

batch_size = 126
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(x_train)
history = classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val),

                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size

                              )
results = classifier.predict(df_test)





results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)