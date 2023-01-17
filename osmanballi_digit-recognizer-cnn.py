# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Train=pd.read_csv("../input/train.csv")

print(Train.info())

print(Train.shape)
Test=pd.read_csv("../input/test.csv")

print(Test.info())

print(Test.shape)
Train.head()

Test.head()
x_Train=Train.drop(labels="label",axis=1)

y_Train=Train["label"]

y_Train.head(10)
plt.figure(figsize=(15,7))

sns.countplot(y_Train,palette="icefire")

plt.title("number of digit classes")

y_Train.value_counts()
img=x_Train.iloc[3].values

img=img.reshape((28,28))

plt.imshow(img,cmap="gray")

plt.title(y_Train[3])

plt.axis("off")

plt.show()
x_Train=x_Train/255

x_Test=Test/255

x_Train=x_Train.values.reshape(-1,28,28,1)

x_Test=x_Test.values.reshape(-1,28,28,1)

print("x train shape",x_Train.shape)

print("x test shape",x_Test.shape)
from keras.utils.np_utils import to_categorical

y_Train=to_categorical(y_Train,num_classes=10)
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(x_Train, y_Train, test_size = 0.1, random_state=2)

print("x_train shape",X_train.shape)

print("x_test shape",X_val.shape)

print("y_train shape",Y_train.shape)

print("y_test shape",Y_val.shape)
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()

#

model.add(Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

#

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

# fully connected

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 15  # for better result increase the epochs

batch_size = 250

datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # dimesion reduction

        rotation_range=0.5,  # randomly rotate images in the range 5 degrees

        zoom_range = 0.5, # Randomly zoom image 5%

        width_shift_range=0.5,  # randomly shift images horizontally 5%

        height_shift_range=0.5,  # randomly shift images vertically 5%

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



datagen.fit(X_train)

history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val), steps_per_epoch=X_train.shape[0] // batch_size)



Y_pred = model.predict(X_val)

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_true = np.argmax(Y_val,axis = 1) 

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

f,ax = plt.subplots(figsize=(8, 8))

sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()

for i in range(9):

    plt.subplot(3, 3, i + 1)

    plt.imshow(X_val[i].reshape(28, 28), cmap='gray', interpolation='none')

    plt.title("predicted class {}".format(Y_pred_classes[i]))

    plt.axis("off")

    plt.show()
plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['Accuracy'], loc='upper left')

plt.show()
plt.plot(history.history['val_loss'],color='red')

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['Loss'], loc='upper left')

plt.show()