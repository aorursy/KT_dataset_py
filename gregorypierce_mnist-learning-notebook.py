# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt # graphing library

%matplotlib inline



# Import tensorflow and keras. We will use the keras that is a part of the tensorflow package

# as opposed to the standalone keras

import tensorflow as tf

from tensorflow import keras

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
training_data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

submission_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
training_data.head()
submission_df.head()
print("Training data (rows, columns):" + str(training_data.shape))

print("Submission data (rows, columns):" + str(submission_df.shape))

# We have a Dataframe that has labelled data of the right shape so let's split up that dataframe

# so that we can use it for test, train, and validation. This allows us to move forward with the data we DO have

#

# Sample the training_data Dataframe and select 70% of the data for the training_df, testing_df, validation_df dataframes

# We set the random_state to a specific number so we can reproduce the results



training_df = training_data.sample(frac=0.7, random_state=1)



# Drop the already selected values from the training_data dataframe, what is left is our testing data

testing_df = training_data.drop( training_df.index )



print("Training data (rows, columns):" + str(training_df.shape))

print("Testing data (rows, columns):" + str(testing_df.shape))



print( str(training_data.shape[0] - (training_df.shape[0] + testing_df.shape[0] )) + " rows not accounted for" )

training_labels = training_df["label"].values

training_df = training_df.drop( columns=['label'])
# Confirm that the data is of the right shape

print( "Training data (rows, columns):" + str(training_df.shape) )

print( "Label Count: " + str(training_labels.size) )
testing_labels = testing_df['label'].values

testing_df = testing_df.drop( columns=['label'])
# Confirm that the data is of the right shape

print( "Testing data (rows, columns):" + str(testing_df.shape) )

print( "Testing Label Count: " + str(testing_labels.size) )

print( training_df.describe() )
# Normalize all the values in the dataframe so that they are values of range 0..255 by dividing the dataframe by 255 (the max value)



testing_df /= 255

training_df /= 255
#reshape the rows into (28x28) since that is the original image format for mnist data

IMG_WIDTH = 28

IMG_HEIGHT = 28

IMG_CHANNELS = 1 #this would be 3 for an RGB image, 4 for RGBA, but just 1 for grayscale image



#reshape the data into image shape

training_values = training_df.values.reshape( training_df.shape[0], IMG_WIDTH, IMG_HEIGHT )
plt.imshow(training_values[0], cmap=plt.get_cmap('gray'))

plt.title( training_labels[0] )
training_values = training_df.values.reshape( training_df.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS )

testing_values = testing_df.values.reshape( testing_df.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS )
print(training_labels.shape)

print(training_labels[:10])
NUM_CLASSES = 10

training_labels_categorical = tf.keras.utils.to_categorical(training_labels)

testing_labels_categorical = tf.keras.utils.to_categorical(testing_labels)



print( training_labels_categorical.shape )

print( testing_labels_categorical.shape )
print( training_labels[0] )

print( training_labels_categorical[0])
import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from tensorflow.python.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint

from tensorflow.python.keras.optimizers import Adam ,RMSprop



print("Tensorflow Version: " + tf.version.VERSION)

print("Keras Version: " + tf.keras.__version__)



# set the number of epochs for training the models

EPOCHS=35



# how many samples will the system see at one time before it updates its weights

BATCH_SIZE=64
model = tf.keras.Sequential()



model.add(layers.Conv2D(32, 3, 3, activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)))

model.add(layers.Conv2D(32, 3, 3, activation='relu'))

model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Dropout(0.25))



model.add(layers.Flatten())

model.add(layers.Dense(128, activation="relu"))

model.add(layers.Dropout(0.50))     

model.add(layers.Dense(10, activation="softmax"))



model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model.summary()        





history = model.fit( training_values, training_labels_categorical, validation_split=0.1, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1 )

plt.plot( history.history['acc'])

plt.plot( history.history['val_acc'])

plt.title( 'Model accuracy')

plt.ylabel( 'accuracy')

plt.xlabel( 'epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()
score = model.evaluate( testing_values, testing_labels_categorical, verbose=1)



print( "Model1 Score: " + str(score) )
model2 = tf.keras.Sequential()



model2.add(layers.Conv2D(32, kernel_size=5, padding='same', activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)))

model2.add(layers.MaxPool2D())

model2.add(layers.Dropout(0.40))



model2.add(layers.Conv2D(64, kernel_size=5, activation='relu'))

model2.add(layers.MaxPool2D())

model2.add(layers.Dropout(0.40))



model2.add(layers.Flatten())

model2.add(layers.Dense(128, activation="relu"))

model2.add(layers.Dropout(0.40))  



model2.add(layers.Dense(10, activation="softmax"))



model2.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



model2.summary()   



history = model2.fit( training_values, training_labels_categorical, validation_split=0.1, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1 )
plt.plot( history.history['acc'])

plt.plot( history.history['val_acc'])

plt.title( 'Model accuracy')

plt.ylabel( 'accuracy')

plt.xlabel( 'epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()
score = model2.evaluate( testing_values, testing_labels_categorical, verbose=1)



print( "Model2 Score: " + str(score) )
model3 = tf.keras.Sequential()



model3.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model3.add(layers.BatchNormalization())



model3.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model3.add(layers.BatchNormalization())



model3.add(layers.MaxPool2D(pool_size=(2,2)))

model3.add(layers.Dropout(0.25))



model3.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model3.add(layers.BatchNormalization())



model3.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model3.add(layers.BatchNormalization())

model3.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model3.add(layers.Dropout(0.25))



model3.add(layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))

model3.add(layers.BatchNormalization())

model3.add(layers.Dropout(0.25))



model3.add(layers.Flatten())

model3.add(layers.Dense(256, activation = "relu"))

model3.add(layers.BatchNormalization())

model3.add(layers.Dropout(0.25))



model3.add(layers.Dense(10, activation = "softmax"))





model3.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model3.summary()



history = model3.fit( training_values, training_labels_categorical, validation_split=0.1, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1 )
plt.plot( history.history['acc'])

plt.plot( history.history['val_acc'])

plt.title( 'Model accuracy')

plt.ylabel( 'accuracy')

plt.xlabel( 'epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()
score = model3.evaluate( testing_values, testing_labels_categorical, verbose=1)



print( "Model3 Score: " + str(score) )
modelCheckPoint = ModelCheckpoint( filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5",

                                   monitor="val_acc",

                                   verbose=1,

                                   save_best_only=True,

                                   mode='max')



# generate more training data

datagen = ImageDataGenerator(

        rotation_range= 8,  

        zoom_range = 0.13,  

        width_shift_range=0.13, 

        height_shift_range=0.13)





initial_learning_rate = 0.001

optimizer = Adam(lr=initial_learning_rate, decay= initial_learning_rate / (EPOCHS*1.3))



model4 = tf.keras.Sequential()



model4.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model4.add(layers.BatchNormalization())



model4.add(layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model4.add(layers.BatchNormalization())



model4.add(layers.MaxPool2D(pool_size=(2,2)))

model4.add(layers.Dropout(0.25))



model4.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model4.add(layers.BatchNormalization())



model4.add(layers.Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model4.add(layers.BatchNormalization())

model4.add(layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))

model4.add(layers.Dropout(0.25))



model4.add(layers.Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same',  activation ='relu'))

model4.add(layers.BatchNormalization())

model4.add(layers.Dropout(0.25))



model4.add(layers.Flatten())

model4.add(layers.Dense(256, activation = "relu"))

model4.add(layers.BatchNormalization())

model4.add(layers.Dropout(0.25))



model4.add(layers.Dense(10, activation = "softmax"))





model4.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])

model4.summary()



history = model4.fit_generator(  datagen.flow(training_values, training_labels_categorical, batch_size=BATCH_SIZE), validation_data=(testing_values, testing_labels_categorical), steps_per_epoch=training_values.shape[0] // BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[modelCheckPoint] )
plt.plot( history.history['acc'])

plt.plot( history.history['val_acc'])

plt.title( 'Model accuracy')

plt.ylabel( 'accuracy')

plt.xlabel( 'epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend( ['train', 'val'], loc='upper left')

plt.show()
score = model4.evaluate( testing_values, testing_labels_categorical, verbose=1)



print( "Score: " + str(score) )
#prepare the submission data

submission_df /= 255

submission_values = submission_df.values.reshape( submission_df.shape[0], IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS )



# model prediction on test data

predictions = model4.predict_classes(submission_values, verbose=0)



# submission

submissions = pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),

    "Label": predictions})

submissions.to_csv("DR.csv", index=False, header=True)