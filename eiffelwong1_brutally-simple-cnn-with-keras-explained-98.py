import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import tensorflow.keras as keras
#reading both files

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

val_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
NUM_CLASS = 10



#making one hot encoding for the label

label = data['label']

label_one_hot = np.zeros((label.size, NUM_CLASS))

for i in range(label.size):

    label_one_hot[i,label[i]] = 1

#remove the label column, so the remaining 784 columns can form a 28*28 photo

del data['label']



#changing data from DataFrame object to a numpy array, cause I know numpy better :p

data = data.to_numpy()

print(data.shape)



#making data to 28*28 photo

data = data.reshape(-1,28,28,1)
#checking out data shape

print(' data shape: {} \n one hot lable shape: {}'.format(

    data.shape, label_one_hot.shape))
#simple CNN model with Keras

import keras

from keras.models import Model, Sequential

from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization

from keras.layers import Dense, Flatten, Activation



model = Sequential([

    

    Convolution2D(filters = 5, kernel_size = (3,3), activation = 'relu', input_shape=(28,28,1)),

    MaxPooling2D(pool_size=(2,2)),

    BatchNormalization(),

    #do a drop out here if interested

    

    Convolution2D(filters = 25, kernel_size = (3,3), activation = 'relu'),

    MaxPooling2D(pool_size=(2,2)),

    BatchNormalization(),

    #do a drop out here if interested

    

    Flatten(),

    Dense(10),

    Activation('softmax')

])
model.compile('adam',

              loss='categorical_crossentropy',

              metrics=['accuracy']

             )
#Starts training the model over the data 10 times.

#Here nothing fancy added for keeping it really really simple.



history = model.fit(data, label_one_hot, epochs = 10, validation_split = 0.1)
import matplotlib.pyplot as plt



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.show()
#we read the csv before, but just read it again here.

val_data = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



#the same way to process the training data after seperating the label

val_data = val_data.to_numpy()

val_data = val_data.reshape(-1,28,28,1)



#here we ask the model to predict what the class is

raw_result = model.predict(val_data)



#note: model.predict will return the confidence level for all 10 class,

#      therefore we want to pick the most confident one and return it as the final prediction

result = np.argmax(raw_result, axis = 1)



#generating the output, remember to submit the result to the competition afterward for your final score.

submission = pd.DataFrame({'ImageId':range(1,len(val_data) + 1), 'Label':np.argmax(raw_result, axis = 1)})

submission.to_csv('SimpleCnnSubmission.csv', index=False)