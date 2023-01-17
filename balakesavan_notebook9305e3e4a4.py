#importing libraries

import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from keras import backend as K

K.set_image_dim_ordering('th')



#for repeatability

#seed = 7

#np.random.seed(seed)



# create data sets, skipping the header row with [1:]

dataset = pd.read_csv("../input/train.csv")

y_train = dataset[[0]].values.ravel()

X_train = dataset.iloc[:,1:].values

test = pd.read_csv("../input/test.csv").values



# reshape to be [samples][pixels][width][height]

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')

test = test.reshape(test.shape[0], 1, 28, 28).astype('float32')



# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

test = test / 255



# one hot encode outputs

y_train = np_utils.to_categorical(y_train)



#subsetting loaded data for use as train and test during initial training and tuning

randIndexTrain = np.random.randint(X_train.shape[0],size=15000)

#randIndexTest = np.random.randint(X_train.shape[0],size=500)



#for use during initial model training/ tuning

X_train1 = X_train[randIndexTrain,:]

y_train1 = y_train[randIndexTrain,:]



#X_test1 = X_train[randIndexTest,:]

#y_test1 = y_train[randIndexTest,:]



#full sized training data load

#X_train1 = X_train

#y_train1 = y_train





#used later to specify the number of neurons in the output layer

num_classes = y_train1.shape[1]





def baseline_model():

	# create model

	model = Sequential()

	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model



def larger_model():

	# create model

	model = Sequential()

	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(15, 3, 3, activation='relu'))

	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Dropout(0.2))

	model.add(Flatten())

	model.add(Dense(128, activation='relu'))

	model.add(Dense(50, activation='relu'))

	model.add(Dense(num_classes, activation='softmax'))

	# Compile model

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

 



# build the model

model = larger_model()



# Fit the model - initial model training and tuning

#model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), nb_epoch=10, batch_size=200, verbose=0)

# Final evaluation of the model

#scores = model.evaluate(X_test1, y_test1, verbose=0)

#print(head(scores))



# save results

#np.savetxt('scores_cnn_keras.csv', np.c_[range(1,len(test)+1),scores], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')



#print("Baseline Error: %.2f%%" % (100-scores[1]*100)) 



#for Kaggle submission



# Fit the model

model.fit(X_train1, y_train1, nb_epoch=10, batch_size=200, verbose=0)



# Prediction

pred = model.predict(test)

pred1 = pred.argmax(axis=0)

pred2 = np.c_[range(1,len(test)+1),pred1]



# save results

np.savetxt('submission_cnn_keras.csv', pred2, delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')