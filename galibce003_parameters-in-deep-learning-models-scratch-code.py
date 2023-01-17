from keras.models import Sequential
from keras.layers import Conv2D, Dense, LSTM
model = Sequential()

#Specify the input size of the first layer
#input_shape = (3, ) - Our input size is 3
#(Dense(5,..... - The number of neuron of our first hiddeen layer will be 5

model.add(Dense(5, activation = 'relu', input_shape = (3, )))

#Output layer

model.add(Dense(2, activation = 'softmax'))
model.summary()
model = Sequential()

#1st hidden layer
model.add(Dense(100, activation = 'relu', input_shape = (50, )))

#2nd hidden layer
model.add(Dense(1, activation = 'relu'))

#3rd hidden layer
model.add(Dense(100, activation = 'relu'))

#Output layer
model.add(Dense(50, activation = 'softmax'))
model.summary()
model = Sequential()

#Here, (3,3) is the kernel size
model.add(Conv2D(1, (3,3), input_shape = (5, 5, 1)))
model.summary()
model = Sequential()

#Here, (3,3) is the kernel size
model.add(Conv2D(1, (3,3), input_shape = (50, 50, 1)))
model.summary()
model = Sequential()

#Here, (3,3) is the kernel size
model.add(Conv2D(1, (3,3), input_shape = (5, 5, 1), use_bias = False))
model.summary()
model = Sequential()

model.add(Conv2D(3, (2,2), input_shape = (5, 5, 1)))

model.summary()
model = Sequential()

#input_shape = (5, 5, 3) - here 3 stands as RGB is used
model.add(Conv2D(1, (2,2), input_shape = (5, 5, 3)))

model.summary()
model = Sequential()

#input_shape = (5, 5, 2) - here 2 stands as RG is used
model.add(Conv2D(3, (2,2), input_shape = (5, 5, 2)))

model.summary()
model = Sequential()

model.add(Conv2D(5, (5,5), input_shape = (5, 5, 3)))

model.summary()
model = Sequential()

model.add(LSTM(units = 2, input_dim = 3, input_length = 6))

model.summary()