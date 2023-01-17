from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras import backend as K

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

import numpy as np



# seed for reproducing same results

seed = 785

np.random.seed(seed)



# load dataset

dataset = np.loadtxt('../input/handwritten-az-dataset/A_Z Handwritten Data.csv', delimiter=',')



# split into input and output variables

X = dataset[:,0:784]

Y = dataset[:,0]



# split the data into training (50%) and testing (50%)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.50, random_state=seed)



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')



X_train = X_train / 255

X_test = X_test / 255



# one hot encode outputs

Y_train = np_utils.to_categorical(Y_train)

Y_test = np_utils.to_categorical(Y_test)



num_classes = Y_test.shape[1]





# create model

model = Sequential()

model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))



# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)



# Final evaluation of the model

scores = model.evaluate(X_test,Y_test, verbose=0)

print("CNN Error: %.2f%%" % (100-scores[1]*100))



model.save('weights.model')