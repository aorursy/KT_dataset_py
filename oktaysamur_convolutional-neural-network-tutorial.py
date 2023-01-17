# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# import warnings

import warnings

# filter warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# train data

train = pd.read_csv("../input/train.csv")

print(train.shape)

train.tail()
#test data

test = pd.read_csv("../input/test.csv")

print(test.shape)

test.tail()
# Putting labels into y_train variable

y_train = train["label"]

#Dropping 'label' column

x_train = train.drop(labels = ["label"],axis = 1)
# visualizing number of digits classses

plt.figure(figsize=(15,7))

h = sns.countplot(y_train, palette="cubehelix")

plt.title("Number of Classes")

y_train.value_counts()
# plot some samples

img = x_train.iloc[0].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[0,0])

plt.axis("off")

plt.show()
# plot some samples

img = x_train.iloc[7].as_matrix()

img = img.reshape((28,28))

plt.imshow(img,cmap='gray')

plt.title(train.iloc[3,0])

plt.axis("off")

plt.show()
#Normalizing data size 

x_train = x_train / 255.0

test = test / 255.0

print("x_train shape:" , x_train.shape)

print("test shape:" , test.shape)
# Reshape

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

print("x_train shape: ",x_train.shape)

print("test shape: ",test.shape)
# Label Encoding 

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

y_train = to_categorical(y_train, num_classes = 10)
# Splitting the train and the validation set for the fitting

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size = 0.1, random_state=42)

print("x_train shape",x_train.shape)

print("x_test shape",x_val.shape)

print("y_train shape",y_train.shape)

print("y_test shape",y_val.shape)
plt.imshow(x_train[4][:,:,0],cmap='gray')

plt.show()
from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam,Nadam,SGD

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

model = Sequential()

# Part 1

model.add(Conv2D(filters = 8,kernel_size = (5,5),padding = 'Same', activation = 'relu', input_shape = (28,28,1)))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

# Part 2 

model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

# Fully Connected

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

# Define the optimizer

optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 100 

batch_size = 250
# data augmentation

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



datagen.fit(x_train)
final = model.fit_generator(datagen.flow(x_train,y_train,batch_size=batch_size),

                           epochs=epochs,

                           validation_data = (x_val,y_val),

                           steps_per_epoch=x_train.shape[0] // batch_size)
# Plotting the loss and accuracy curves for training and validation 

plt.plot(final.history['val_loss'], color='b', label="validation loss")

plt.title("Test Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
# confusion matrix

#Predicting the values from the validation dataset

y_pred = model.predict(x_val)

# converting predictions classes to one hot vectors

y_pred_classes = np.argmax(y_pred,axis=1)

#converting validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1)

# compute the confusion matrix

confusion = confusion_matrix(y_true, y_pred_classes)

# plotting

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(confusion, annot=True,linewidths=0.01,cmap="Reds",linecolor="gray", fmt= '.1f', ax=ax)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

plt.title("Confusion Matrix")

plt.show()