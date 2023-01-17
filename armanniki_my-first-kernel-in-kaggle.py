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
# Loading Data

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



y_train = train["label"]

x_train = train.drop(labels = ["label"], axis=1)



del train



#g = sns.countplot(y_train)

#y_train.value_counts()
# Normalizing Data

x_train = x_train / 255.0

test = test / 255.0

#Reshaping images in 3 dimensions

x_train = x_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

from keras.utils.np_utils import to_categorical



y_train = to_categorical(y_train, num_classes = 10)

#Spliting data into train and validation set

from sklearn.model_selection import train_test_split



random_seed = 2

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)

#g = plt.imshow(x_train[5][:,:,0])
#CNN Model

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.layers.normalization import BatchNormalization





model = Sequential()



model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=3,activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))



model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.4))

model.add(Dense(10, activation='softmax'))



model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
epochs = 60 # but actually it should be more than 1

batch_size = 86

#Data augmentation

from keras.preprocessing.image import ImageDataGenerator



datageneration = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False, 

        zca_whitening=False,  

        rotation_range=10, 

        zoom_range = 0.1,

        width_shift_range=0.1, 

        height_shift_range=0.1,  

        horizontal_flip=False,  

        vertical_flip=False)  



datageneration.fit(x_train)

#fitting the model

history = model.fit_generator(datageneration.flow(x_train,y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (x_val,y_val),

                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size)

# Confusion matrix 

import numpy as np

from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

y_pred = model.predict(x_val)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_val,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
model.save("dig_rec.model")
from keras.models import model_from_json



# serialize model to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")
# predict results

results = model.predict(test)



# select the indix with the maximum probability

results = np.argmax(results,axis = 1)



results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)



submission.to_csv("cnn_mnist_datagen.csv",index=False)