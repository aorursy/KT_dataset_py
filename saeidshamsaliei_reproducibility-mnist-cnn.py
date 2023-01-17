import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, MaxPooling2D, Conv2D, Flatten

import numpy as np

import pandas as pd
batch_size = 128

num_epoch = 25



#load mnist dataset

#(X_train, y_train), (X_test, y_test) = mnist.load_data() 



a = np.load("../input/keras-mnist-amazonaws-npz-datasets/mnist.npz")

X_test = a['x_test']

y_test = a['y_test']



X_train = a['x_train']

y_train = a['y_train']



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

input_shape = (28, 28, 1)
#more reshaping

X_train = X_train.astype('float32')

X_test = X_test.astype('float32')

X_train /= 255

X_test /= 255

print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)



#set number of categories

num_category = 10

# convert class vectors to binary class matrices

y_train = keras.utils.to_categorical(y_train, num_category)

y_test = keras.utils.to_categorical(y_test, num_category)
##model building

model = Sequential()

#convolutional layer with rectified linear unit activation

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

#32 convolution filters used each of size 3x3

#choose the best features via pooling

model.add(MaxPooling2D(pool_size=(2, 2)))



#again

model.add(Conv2D(64, (3, 3), activation='relu'))

#64 convolution filters used each of size 3x3

#choose the best features via pooling

model.add(MaxPooling2D(pool_size=(2, 2)))

#randomly turn neurons on and off to improve convergence





model.add(Dropout(0.25))

#flatten since too many dimensions, we only want a classification output

model.add(Flatten())

#fully connected to get all relevant data

model.add(Dense(128, activation='relu'))

#one more dropout for convergence' sake  

model.add(Dropout(0.5))

#output a softmax to squash the matrix into output probabilities

model.add(Dense(num_category, activation='softmax'))





#We use adam as our optimizer

#categorical ce since we have multiple classes (10) 

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer="adam",

              metrics=['accuracy'])





#model training

model_log = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch,

          verbose=1, validation_data=(X_test, y_test))



score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0]) #Test loss: 0.0296396646054

print('Test accuracy:', score[1]) #Test accuracy: 0.9904





#https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
#Save the classifications to a .csv file called results.csv

Y_predicted = model.predict(X_test)

pred_label = Y_predicted.argmax(axis = 1)

image_id = range(1,len(Y_predicted)+1)

df = {'ImageId':image_id,'Label':pred_label}

df = pd.DataFrame(df)

df.to_csv('results.csv',index = False)



#Save the model as a HDF5 file called model.h5

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'