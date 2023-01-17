import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#model building modules

from keras.models import Sequential # initial instance of model

from keras.layers import Dense # layers 

from keras.utils import np_utils #OneHotEncoding
train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv') # load train

test = pd.read_csv('../input/mnist-in-csv/mnist_test.csv') # load test
train.head(1) # train head
test.head(1) # test head
print(train.info())

print(test.info())
def lab(df):

    lab_dum = np_utils.to_categorical(df['label']) # convert label to dummy variable

    return lab_dum
y_train = lab(train)

y_test = lab(test)
X_train = train.iloc[:,1:] # create X_train

X_test = test.iloc[:,1:]# create X_test



# normalize X Train and X test as they are between 0 and 255

X_train /= 255

X_test /= 255
X_train.info()
# create a function to instanciate, add layers, compile the model

def nnmod():

	# instanciate and add layers to model

	nnmod = Sequential()

	nnmod.add(Dense(784, input_dim=784, activation='relu'))

	nnmod.add(Dense(10, activation='softmax'))

	# compile model

	nnmod.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	return nnmod
nnmod = baseline_model() # run our function



nnmod.fit(X_train, y_train, batch_size=200, epochs=10, validation_data=(X_test, y_test)) # fit model
accuracy = nnmod.evaluate(X_test, y_test)

print("Accuracy is: ", score[1]*100, "%")