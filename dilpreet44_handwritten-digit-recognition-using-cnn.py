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

import seaborn as sea

import matplotlib.image as img

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau
train_df = pd.read_csv('../input/digit-recognizer/train.csv')

test_df  = pd.read_csv('../input/digit-recognizer/test.csv')
y_train = train_df['label']

x_train = train_df.drop(['label'],axis=1)

plot1 = sea.countplot(y_train)

y_train.value_counts()
train_df.shape
x_train.isnull().any().describe()
x_train = x_train/255

test_df = test_df/255
x_train = x_train.values.reshape(-1,28,28,1)
test_df = test_df.values.reshape(-1,28,28,1)
y_train = to_categorical(y_train, num_classes = 10)
y_train
train_X,test_X,train_Y,test_Y = train_test_split(x_train,y_train,test_size = 0.2,random_state=20)
plt.imshow(train_X[1][:,:,0])
model = Sequential()
# CNN

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(28,28,1)))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.25))



model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))

model.add(Conv2D(128,kernel_size=(3,3),activation='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss = 'categorical_crossentropy',metrics=['accuracy'])
model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit(train_X, train_Y,batch_size=120,epochs=150,verbose=1)

score = model.evaluate(test_X, test_Y, verbose=0)
score
results = model.predict(test_df)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("my_submission.csv",index=False)