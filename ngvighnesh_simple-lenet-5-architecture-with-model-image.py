import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import all useful libraries

#Data Processing libraries

import numpy as np

import pandas as pd

from keras.preprocessing.image import ImageDataGenerator               # used for data augmentation

#ML Libraries

import tensorflow as tf

import keras 

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Conv2D, AveragePooling2D, Flatten, Dropout
# read train and test data

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
#inspect the shape of the dataset

print(train.shape)

print(test.shape)

# 784 = pixels of a 28x28 image

# 785 = pixels of a 28x28 image + class of the data

# The large dimention is the dimention with no of examples
# split into x and y

y_train = train['label']

x_train = train.drop(labels = ['label'],axis = 1)



#clear up the memory

del train



# print the y_train which has the class of each data

y_train
# scale values to between 0 and 1 for faster learning

x_train = x_train/255



# you want number of imput channels to be last index for this version of keras

image_size = int(np.sqrt(x_train.shape[1]))



ip_shape = (image_size, image_size, 1)

x_train = x_train.values.reshape(x_train.shape[0], image_size, image_size, 1)



# convert y to one hot vectors for training

y_train = keras.utils.np_utils.to_categorical(y_train.values, num_classes=10)
y_train
print('x_train.shape = ', x_train.shape)

print('y_train.shape = ', y_train.shape)



# 42000 = number of examples

# 28    = no of pixels

# 1     = number of channels ( the image is black/white so has only one channel)

# 10    = number of classes
# process the test model similarly

test = test/255

test = test.values.reshape(test.shape[0], image_size, image_size, 1)



print(test.shape)
# Keep aside a part of the training set (10000 examples ) for development

x_train, x_dev, y_train, y_dev = train_test_split(x_train, y_train, test_size=10000, random_state = 12)
#inspect the shape to make sure the train dev split was successful

print('x_train.shape = ', x_train.shape)

print('y_train.shape = ', y_train.shape)

print('x_dev.shape   = ', x_dev.shape)

print('y_dev.shape   = ', y_dev.shape)
# build a keras model

model = keras.Sequential()

model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=ip_shape ))

model.add(AveragePooling2D())

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))

model.add(AveragePooling2D())

model.add(Flatten())

model.add(Dense(units=120, activation='relu'))

model.add(Dense(units=84, activation='relu'))

model.add(Dense(units=10, activation = 'softmax'))

# even though the original paper did not use relu we'll use it as it is better.

# The reason why relu wasn't used on the og paper was because it was not famous at the time of writing the paper
# Summary of the model descriing it's structure

model.summary()
# compile the model with a loss function and an optimizer

model.compile(loss = keras.losses.categorical_crossentropy,

             optimizer = keras.optimizers.Adam(),

             metrics = ['accuracy'])
# train the model for 15 epochs using batches of size 128

model.fit(x_train, y_train, batch_size = 128, epochs = 15)
# crosscheck with dev set for overfitting

dev_loss, dev_metric = model.evaluate(x_dev, y_dev)

print('Accuracy = ', dev_metric)
# predict the results 

results = model.predict(test)

results = np.argmax(results,axis = 1)

results
#convert the resuts into a dataframs with the appropriate ImageId

results_df = pd.DataFrame()

results_df['ImageId'] = np.arange(len(results)) + 1

results_df['Label'] = pd.Series(results)

results_df
#save the results as a dataframe for submission

results_df.to_csv('submission.csv', index = False)