# import all of our packages

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.layers import Activation, Dense, Dropout, Flatten

from keras.layers import Conv2D, MaxPool2D

from keras.utils import np_utils 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.callbacks import ReduceLROnPlateau
# import our dataset using the pandas read_csv option

test_import = pd.read_csv("../input/Kannada-MNIST/test.csv");

train_import = pd.read_csv("../input/Kannada-MNIST/train.csv");
# test the data for null values. notice how the functions are stacked. isnull() shows a 1 if its null. Any() checks which ones. 

# Describe shows it in a format thats easier to read (otherwise you would have to check the whole matrix if theres a 1 somewhere)

train_import.isnull().any().describe()
# do this again for the training set

train_import.isnull().any().describe()
# drop the label column from the dataset so we only keep the pixel information. The first column represents

# the outcome (number between 0 and 10).

X_train = train_import.drop('label', axis=1)

# obtain the first column vector from the dataset and use it to label data as this is the outcome. (number between 0 and 10)

y_train = train_import.iloc[:, 0]
# check the histogram of distribution from our training set using the built in function hist(). Alternatively you could

# use a library such as matplotlib.  We want to make sure that the distribution between numbers that we have examples of are

# the same. Imagine if we had 1000 examples of the number 2 but only a couple for the number 3. Its just a quick check.

y_train.hist()
test_import = test_import.drop('id', axis=1)
# normalize the pixel intensity values. Great video on normalization: https://www.youtube.com/watch?v=FDCfw-YqWTE

X_train = X_train / 255.0

test_import = test_import / 255.0



# reshape it from a m * 728 matrix (m is number of examples) to a matrix of m * 28 * 28 where 1 is an additional channel.

# usually you use this last channel for the RGB values but in this example its not needed

X_train = X_train.values.reshape(-1,28,28,1)

test = test_import.values.reshape(-1,28,28,1)



# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

# reason for this is that in a neural network each node in the output layer outputs 0 or 1.

# more information about one hot encoder

# https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f

Y_train =  np_utils.to_categorical(y_train, num_classes = 10)



# split set into training and validation. 

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
# creating the model. Guide from Keras: https://keras.io/getting-started/sequential-model-guide/

# more information about the activation functions relu vs softmax: https://github.com/Kulbear/deep-learning-nano-foundation/wiki/ReLU-and-Softmax-Activation-Functions

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28,28,1), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))

model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(10, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# create more sample images using ImageDataGenerator.This ensures we have more data to train on. More info about function here: https://keras.io/preprocessing/image/

imagegen = ImageDataGenerator(

        featurewise_center=False, 

        samplewise_center=False, 

        featurewise_std_normalization=False, 

        samplewise_std_normalization=False,

        zca_whitening=False, 

        rotation_range=9, 

        zoom_range = 0.25,

        width_shift_range=0.25,

        height_shift_range=0.25, 

        horizontal_flip=False, 

        vertical_flip=False)



imagegen.fit(X_train)
# set epochs (1 epoch = 1 run) higher for better result (set lower to save costs)

epochs = 4

batch_size = 64
# fit the model with generated images. Asign it to fitobj so we can later check on the data

fitobj = model.fit_generator(imagegen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 1, steps_per_epoch=X_train.shape[0])
score = model.evaluate(X_val, Y_val, batch_size=30)
# review the model score

score
# view history of model. It prints for every epoch so you can graph it out to see if you are actually improving

fitobj.history



# highly recommend to review your data afterward to see which examples were false positives but let's stick to the bare minimum

# to get this kernel to run.
# predict results

results = model.predict(test)

results = np.argmax(results,axis = 1)

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = results

submission.to_csv('submission.csv', index=False)

submission.head()
# check if we have a bias towards a specific example in the results

submission.hist()