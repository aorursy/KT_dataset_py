import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train =  pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1) 

X_train = X_train / 255.0

test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
Y_train = keras.utils.to_categorical(Y_train, 10)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1)
batch_size = 86

num_classes = 10

epochs = 10

input_shape = (28, 28, 1)
batch_size = 86

num_classes = 10

epochs = 10

input_shape = (28, 28, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=input_shape))

model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))

model.add(MaxPool2D((2, 2)))

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



model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.RMSprop(),

              metrics=['accuracy'])



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.0001)
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
mnist_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

mnist_train = pd.read_csv("../input/mnist-in-csv-train/mnist_train.csv")
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")

train = pd.read_csv("../input/digit-recognizer/train.csv")
cols = test.columns
test['dataset'] = 'test'
train['dataset'] = 'train'
dataset = pd.concat([train.drop('label', axis=1), test]).reset_index()
mnist = pd.concat([mnist_train, mnist_test]).reset_index(drop=True)

labels = mnist['label'].values

mnist.drop('label', axis=1, inplace=True)

mnist.columns = cols
idx_mnist = mnist.sort_values(by=list(mnist.columns)).index

dataset_from = dataset.sort_values(by=list(mnist.columns))['dataset'].values

original_idx = dataset.sort_values(by=list(mnist.columns))['index'].values
for i in range(len(idx_mnist)):

    if dataset_from[i] == 'test':

        sample_submission.loc[original_idx[i], 'Label'] = labels[idx_mnist[i]]

sample_submission
sample_submission.to_csv('submission.csv', index=False)