import numpy as np 

import pandas as pd 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import numpy

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers.convolutional import Convolution2D

from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils

from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt

from matplotlib import cm
seed = 18

numpy.random.seed(seed)



train = pd.read_csv("../input/train.csv")

target = train["label"]



y_train = train["label"]

X_train = train.drop("label",1)



y_train = y_train[2000:]

y_valid = y_train[:2000]



X_train = X_train[2000:]

X_valid = X_train[:2000]
X_train = StandardScaler().fit(X_train).transform(X_train)

X_valid = StandardScaler().fit(X_valid).transform(X_valid)



X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')

X_valid = X_valid.reshape(X_valid.shape[0], 28, 28, 1).astype('float32')



for i in range(9):

    plt.subplot(331+i)

    plt.imshow(X_train.reshape(-1,1,28,28)[i][0], cmap=cm.binary)

plt.show()

print(target[2000:2009])
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255

X_valid = X_valid / 255

# one hot encode outputs

y_train = np_utils.to_categorical(y_train)

y_valid = np_utils.to_categorical(y_valid)

num_classes = y_valid.shape[1]
def larger_model():

	# create model

	model = Sequential()

	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(28, 28, 1), activation='relu'))

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

# Fit the model

model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=25, batch_size=200, verbose=2)

# Final evaluation of the model

scores = model.evaluate(X_valid, y_valid, verbose=0)

print("Classification Error: %.2f%%" % (100-scores[1]*100))
test = pd.read_csv("../input/test.csv")

test = StandardScaler().fit(test).transform(test)

test = test.reshape(test.shape[0], 28, 28, 1).astype('float32')

test = test / 255
submission = model.predict_classes(test, verbose=2)
pd.DataFrame({"ImageId": list(range(1,len(test)+1)), 

              "Label": submission}).to_csv('submission.csv', index=False,header=True)