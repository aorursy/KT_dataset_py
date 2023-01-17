# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from matplotlib.pyplot import imshow # For showing some samples from the data

from matplotlib.pyplot import axis

from matplotlib.pyplot import show





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
y = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy')

x = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy')
print("Shape of x dataset is: ",x.shape)

print("Shape of y dataset is:",y.shape)
print(x[0])

print("--------------------------------------------------------------------------")

print(y[0])
#Slicing one sample

random_sample = x[0]

imshow(random_sample)

axis('off')

show()
#Slicing one sample

random_sample = x[334]

imshow(random_sample)

axis('off')

show()
#Slicing one sample

random_sample = x[765]

imshow(random_sample)

axis('off')

show()
from keras.models import Sequential

from keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1) # Train Test Splitting
x_train = x_train.reshape((-1,64,64,1))

x_test = x_test.reshape((-1,64,64,1))

print("Shape of x_train is now: ",x_train.shape)

print("Shape of x_test is now: ",x_test.shape)
model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu",input_shape=(64,64,1))) #Convolutional Operator

model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu")) #Convolutional Operator



model.add(MaxPool2D(pool_size=(5,5))) # Max Pool

model.add(Dropout(0.25)) # Dropout



model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu"))

model.add(Conv2D(filters=32,kernel_size=(5,5),padding="Same",activation="relu"))



model.add(MaxPool2D(pool_size=(5,5)))

model.add(Dropout(0.25))



model.add(Flatten())



model.add(Dense(256,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(256,activation="relu"))

model.add(Dropout(0.25))

model.add(Dense(10,activation="softmax"))

# There are hyperparameters

optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)



model.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,

          y_train,

          batch_size=30,

          epochs=25)
from sklearn.metrics import accuracy_score

from keras.utils.np_utils import to_categorical



y_head = model.predict_classes(x_test)

y_head = to_categorical(y_head,num_classes=10)

print(y_head.shape)

print(y_test.shape)
print("Accuracy of model is ",accuracy_score(y_test,y_head))