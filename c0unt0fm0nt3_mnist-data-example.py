# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

from IPython.display import HTML

from keras.datasets import mnist

from keras.utils import to_categorical



def create_download_link(title = "Download CSV file", filename = "data.csv"):  

    html = '<a href={filename}>{title}</a>'

    html = html.format(title=title,filename=filename)

    return HTML(html)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from keras import layers

from keras import models



model=models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation='relu'))



model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])

import pandas as pd

train_file_name="/kaggle/input/digit-recognizer/train.csv"

test_file_name="/kaggle/input/digit-recognizer/test.csv"

df_train=pd.read_csv(train_file_name)

pixel_list=df_train.columns.values[1:]

train_labels=df_train.label.values

train_images=df_train[pixel_list]

train_images=train_images.values.reshape((len(df_train),28,28))

train_images=train_images.reshape((len(df_train),28,28,1))

train_images=train_images.astype('float32')/255



df_test=pd.read_csv(test_file_name)

pixel_list=df_test.columns.values

test_images=df_test[pixel_list]

test_images=test_images.values.reshape((len(df_test),28,28))

test_images=test_images.reshape((len(df_test),28,28,1))

test_images=test_images.astype('float32')/255



train_labels=to_categorical(train_labels)
model.fit(train_images, train_labels, epochs=5, batch_size=64)
predicitions=model.predict(test_images)

df_predicitions=pd.DataFrame({"ImageId":np.arange(1,len(predicitions)+1),"Label":predicitions.argmax(axis=1)})

# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='submission.csv')
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    horizontal_flip=False)

# fit parameters from data

datagen=datagen.flow(train_images, train_labels)
'''

steps_per_epoch: Integer.

                Total number of steps (batches of samples)

                to yield from `generator` before declaring one epoch

                finished and starting the next epoch. It should typically

                be equal to the number of samples of your dataset

                divided by the batch size.

                Optional for `Sequence`: if unspecified, will use

                the `len(generator)` as a number of steps.

'''             
len(train_images)
# fits the model on batches with real-time data augmentation:

model.fit_generator(datagen,

                    steps_per_epoch=len(train_images) / 32, epochs=10)
predicitions=model.predict(test_images)

df_predicitions=pd.DataFrame({"ImageId":np.arange(1,len(predicitions)+1),"Label":predicitions.argmax(axis=1)})



# create a link to download the dataframe which was saved with .to_csv method

create_download_link(filename='submission.csv')
for data_batch, labels_batch in datagen:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break

    
history=model.fit_generator(datagen,epochs=10)
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline

np.random.seed(2)

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator



from keras.callbacks import ReduceLROnPlateau



sns.set(style='white', context='notebook', palette='deep')

train_file_name="/kaggle/input/digit-recognizer/train.csv"

test_file_name="/kaggle/input/digit-recognizer/test.csv"



train=pd.read_csv(train_file_name)

test=pd.read_csv(test_file_name)



Y_train=train['label']

X_train=train.drop(labels=['label'], axis=1)



del train

g=sns.countplot(Y_train)

Y_train.value_counts()
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train=X_train/255.0

test=test/255.0
X_train=X_train.values.reshape(-1,28,28,1)

test=test.values.reshape(-1,28,28,1)

Y_train=to_categorical(Y_train, num_classes=10)
random_seed=2
X_train, X_val, Y_train, Y_val=train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)
g=plt.imshow(X_train[0][:,:,0])
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
# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
# Set a learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
epochs = 10 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 86
# With data augmentation to prevent overfitting (accuracy 0.99286)



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





datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

                              , callbacks=[learning_rate_reduction])
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['acc'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)