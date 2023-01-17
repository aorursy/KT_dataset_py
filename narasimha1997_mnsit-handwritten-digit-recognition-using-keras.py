from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.datasets import mnist
import numpy
from keras.utils import np_utils


(X, Y), (test_x, test_y) = mnist.load_data()

X = X.reshape((X.shape[0], 28, 28, 1))

test_x.reshape((test_x.shape[0], 28, 28, 1))

print(X.shape, test_x.shape)

#create categorical sets:
Y = np_utils.to_categorical(Y, num_classes = 10)

#build a simple categotical neural CNN:

def CNN():

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), strides = (1, 1), activation = 'relu', input_shape = (28, 28, 3)))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Conv2D(32, kernel_size = (3,3), strides = (1, 1), activation = 'relu'))
    model.add(Flatten())

    #Dense neural net:
    model.add(Dense(units = 20, activation = 'relu'))
    model.add(Dense(units = 10, activation = 'softmax'))

    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    return model

model = CNN()
model.fit(X, Y, epochs = 20, verbose = 1)
model.save('Model.h5')
