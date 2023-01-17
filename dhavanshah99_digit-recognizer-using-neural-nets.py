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
df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

no_of_rows = len(df_train)

no_of_columns_pixels = len(df_train.columns) - 1

no_of_labels = len(set(df_train['label']))

print("No. of rows = " + str(no_of_rows))

print("No. of columns = " + str(no_of_columns_pixels))

print("No. of labels = " + str(no_of_labels))

X = df_train.iloc[:,1:]

Y = df_train.iloc[:,0]
X = X.values.reshape(-1,28,28,1)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0 )
from keras.models import Sequential

from keras.layers import Dense

from keras.layers.convolutional import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten
#CNN

classifier = Sequential()
classifier.add(Conv2D(32,(3,3),input_shape = (28,28,1),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(256,activation = 'relu')) 
classifier.add(Dense(10,activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255, zoom_range = 0.2, horizontal_flip = True)
train_datagen.fit(X_train)
history = classifier.fit_generator(train_datagen.flow(X_train, Y_train, batch_size = 512), epochs = 40)
test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test = test.values.reshape(-1,28,28,1)
results = classifier.predict(test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST-CNN.csv",index=False)
results
submission