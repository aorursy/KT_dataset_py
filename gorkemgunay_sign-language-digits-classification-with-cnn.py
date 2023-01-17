# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.metrics import confusion_matrix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# laod data from numpy file

X = np.load('../input/sign-language-digits-dataset/X.npy')

y = np.load('../input/sign-language-digits-dataset/Y.npy')

# reshape X

X = X.reshape(-1,64,64,1)



print("X Shape:",X.shape)

print("Y Shape:",y.shape)
plt.figure(figsize=(20,6))



for i,j in enumerate([0,205,411,617,823,1030,1237,1444,1650,1858]):

    plt.subplot(2,5,i+1)

    plt.subplots_adjust(top = 2, bottom = 1)

    plt.imshow(X[j].reshape(64,64))

    plt.title(np.argmax(y[j]))

    plt.axis('off')

list_y = []

list_y = [np.where(i == 1)[0][0] for i in y]

count = pd.Series(list_y).value_counts()

print(count)
plt.figure(figsize = (10,5))

sns.countplot(np.array(list_y))

plt.show()
X_organized = np.concatenate((X[204:409,:],

                              X[822:1028,:],

                              X[1649:1855,:],

                              X[1443:1649,:],

                              X[1236:1443,:],

                              X[1855:2062,:],

                              X[615:822,:],

                              X[409:615,:],

                              X[1028:1236,:],

                              X[0:204,:]),axis = 0)

plt.figure(figsize=(20,6))



for i,j in enumerate([0,205,411,617,823,1030,1237,1444,1650,1858]):

    plt.subplot(2,5,i+1)

    plt.subplots_adjust(top = 2, bottom = 1)

    plt.imshow(X_organized[j].reshape(64,64))

    plt.title(np.argmax(y[j]))

    plt.axis('off')
from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(X_organized,y,test_size = 0.2,random_state = 42)



print("x_train shape:",x_train.shape)

print("x_test shape:",x_test.shape)

print("y_train shape:",y_train.shape)

print("y_test shape:",y_test.shape)
from keras.preprocessing.image import ImageDataGenerator
def show_new_samples(new_images):

    plt.figure(figsize=(20,6))

    for i in range(10):

        plt.subplot(2,5,i+1)

        image = new_images.next()

        plt.imshow(image[0].reshape(64,64))

        plt.axis('off')

    

    plt.show()
#Changin zoom level

datagen = ImageDataGenerator(zoom_range = 0.5)

new_images = datagen.flow(x_train,batch_size = 250)

show_new_samples(new_images)
# Changing rotaion 

datagen = ImageDataGenerator(rotation_range = 45)

new_images = datagen.flow(x_train,batch_size = 250)

show_new_samples(new_images)
# Changing rotaion, zoom 

datagen = ImageDataGenerator(zoom_range = 0.5,rotation_range = 45)

new_images = datagen.flow(x_train,batch_size = 1)

show_new_samples(new_images)
# load libraries

from keras.models import Sequential

from keras.layers import Dense, Dropout, MaxPool2D, Conv2D, Flatten

from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (9,9),padding = 'Same', activation ='relu', input_shape = (64,64,1)))

model.add(MaxPool2D(pool_size=(5,5)))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 64, kernel_size = (7,7),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(4,4), strides=(3,3)))

model.add(Dropout(0.2))



model.add(Conv2D(filters = 128 , kernel_size = (5,5),padding = 'Same',activation ='relu'))

model.add(MaxPool2D(pool_size=(3,3), strides=(2,2)))

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dropout(0.2))

model.add(Dense(256, activation = "relu"))

model.add(Dense(10, activation='softmax'))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)



model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



datagen = ImageDataGenerator(zoom_range = 0.5,rotation_range = 45)

datagen.fit(x_train)



history = model.fit(datagen.flow(x_train,y_train, batch_size=250),epochs = 100, validation_data = (x_test,y_test))
plt.figure(figsize = (10,5))

plt.plot(history.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
y_predict = model.predict(x_test)

y_predict_classes = np.argmax(y_predict,axis = 1) 

y_true = np.argmax(y_test,axis = 1) 

confusion_mtx = confusion_matrix(y_true, y_predict_classes) 

plt.figure(figsize = (10,10))

sns.heatmap(confusion_mtx, annot=True,fmt= '.1f')

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()