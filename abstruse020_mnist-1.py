# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import random



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.image as mpimg

from matplotlib import pyplot

%matplotlib inline







from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix

import itertools
Train = pd.read_csv("../input/train.csv")

Test = pd.read_csv("../input/test.csv")
Train.head()

Train.shape

y_train = Train['label']



X_train = Train.drop(labels='label', axis=1)

y_train.value_counts()
X_train.isnull().any().describe()

Test.isnull().any().describe()
#normalizing the data

X_train = X_train/255.0

Test = Test/255.0
#reshaping the data

X_train = X_train.values.reshape(-1,28,28,1)

Test = Test.values.reshape(-1,28,28,1)

print(Test.shape,X_train.shape)
#converting the label to one hot encoded vector

y_train = to_categorical(y_train, num_classes = 10)
#spliting the data into test and validate

X_train,X_val,Y_train,Y_val =train_test_split(X_train, y_train, test_size=0.1, random_state=0)
print(X_train.shape,X_val.shape,Y_train.shape,Y_val.shape)
plt.imshow(X_train[0][:,:,0])
#image augmentation

datagen = ImageDataGenerator( rotation_range = 15, width_shift_range = 0.1, height_shift_range = 0.1)



datagen.fit(X_train)

np.concatenate((X_train,X_train),axis=0)

random.seed(12345)

for X_batch, Y_batch in datagen.flow(np.concatenate((X_train,X_train),axis=0), np.concatenate((Y_train,Y_train),axis=0), batch_size=35700):

    break

X_train_aug = X_batch

Y_train_aug = Y_batch





for i in range(0, 9):

        pyplot.subplot(330 + 1 + i)

        pyplot.imshow(X_train_aug[i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))

# show the plot

pyplot.show()
#combining the data

X_train_aug = np.concatenate((X_train,X_train_aug),axis=0)

Y_train_aug = np.concatenate((Y_train,Y_train_aug),axis=0)

print(X_train_aug.shape,Y_train_aug.shape)
#CNN architecture => (Conv2D(ReLU)--Conv2D(ReLU)--MaxPool2D--Dropout)*2--Flatten--Dense--Droupout--Out



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
#optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#compiling the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 10

batch_size = 86
#compiling the model

history = model.fit(X_train_aug,

                    Y_train_aug,

                    batch_size = batch_size,

                    epochs = epochs,

                    validation_data = (X_val, Y_val),

                    verbose = 2,

                   callbacks=[learning_rate_reduction])
fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Train loss")

ax[0].plot(history.history['val_loss'], color='r', label="Val loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Train accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Val accuracy")

legend = ax[1].legend(loc='best', shadow=True)
result = model.predict(Test)

result = np.argmax(result,axis = 1)



result.shape
results = pd.Series(result,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)