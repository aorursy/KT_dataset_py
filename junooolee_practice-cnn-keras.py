# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline





import itertools

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from tensorflow import keras

import tensorflow as tf
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.shape , test.shape
train.head()
x_train = train.drop(["label"],axis=1)

y_train = train["label"]



sns.countplot(y_train)

y_train.value_counts()
x_train.isnull().any().describe()
test.isnull().any().describe()
x_train.describe()
x_train = np.array(x_train)

test = np.array(test)
x_train = x_train.reshape(-1,28,28,1)

test = test.reshape(-1,28,28,1)
y_train = to_categorical(y_train,10)
x_train, x_val, y_train, y_val=train_test_split(x_train,y_train,test_size=0.1)
plt.imshow(x_train[0][:,:,0]), y_train[0]
model = Sequential()
x_train.shape,x_val.shape
# input

model.add(Input(shape=(28, 28, 1)))



model.add(Conv2D(filters=64, kernel_size =(3,3),padding = 'Same', activation ='relu'))

model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Conv2D(filters=64, kernel_size =(3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(1,1)))

model.add(Conv2D(filters=64, kernel_size =(3,3),padding = 'Same',activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Dropout(0.25))



model.add(Conv2D(128, (2,2),padding = 'Same', activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (2,2),padding = 'Same', activation ='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))





model.add(Dropout(0.25))





#flatten

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))



#output

model.add(Dense(10, activation='softmax'))
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,\

              optimizer = tf.keras.optimizers.Adam(),\

              metrics=['accuracy'])
model.fit(x_train,y_train, batch_size = 128, epochs = 30, validation_data=(x_val,y_val))
# Predict the values from the validation dataset

y_pred = model.predict(x_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_pred_classes

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1)

y_true

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

confusion_mtx
model.evaluate(x_val,y_val,verbose=0)
train_image_generator = ImageDataGenerator()
history = model.fit_generator(train_image_generator.flow(x_train,y_train, batch_size =32),epochs = 3,validation_data = (x_val,y_val))
# Predict the values from the validation dataset

y_pred = model.predict(x_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

y_pred_classes
# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1)

y_true
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

confusion_mtx
model.summary()
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)