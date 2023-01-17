import pandas as pd

import numpy as np

#loading the data

train = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')

test = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')

print(train.shape)
Y_train = train.iloc[:,0]

Y_train = pd.DataFrame(Y_train).to_numpy()

Y_train = Y_train.reshape(60000)

X_train = train.iloc[:,1:785]

X_train = pd.DataFrame(X_train).to_numpy()

Y_test = test.iloc[:,0]

Y_test = pd.DataFrame(Y_test).to_numpy()

Y_test = Y_test.reshape(10000)

X_test = test.iloc[:,1:785]

X_test = pd.DataFrame(X_test).to_numpy()
X_train = X_train.reshape(60000,28,28)

X_test = X_test.reshape(10000,28,28)
#we want pixels to be between 0 and 1

X_train = X_train/255

X_test = X_test/255
import matplotlib.pyplot as plt

%matplotlib inline



def show_number(indeks):

    """Show number using matplotlib, indeks should be in range 0 - 59999"""

    plt.imshow(X_train[indeks])

    plt.title("Digit: " + str(Y_train[indeks]))

    plt.show()

    

show_number(5)

X_train = X_train.reshape(60000,784)

X_test = X_test.reshape(10000,784)
def examine_model(model, X_train, Y_train, X_test, Y_test):

    model.fit(X_train,Y_train)

    accuracy = model.score(X_test, Y_test)

    return accuracy
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier
accuracy = examine_model(LogisticRegression(max_iter=1000), X_train, Y_train, X_test, Y_test)

print("Logistic regression:")

print(accuracy)
accuracy = examine_model(RandomForestClassifier(), X_train, Y_train, X_test, Y_test)

print("Random Forest Classifier:")

print(accuracy)
accuracy = examine_model(KNeighborsClassifier(),X_train, Y_train, X_test, Y_test)

print("K-Neighbors Classifier:")

print(accuracy)
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D

from keras.layers import AveragePooling2D, MaxPooling2D

from keras.models import Model
def two_layers(input_shape):

    """The model of the neural network with two hidden fully connected layers and softmax output"""

    X_input = Input(input_shape)

    X = Dense(748, activation="relu", name="first")(X_input)

    X = Dense(748, activation="relu", name="second")(X)

    X = X = Dense(10, activation="softmax", name="last")(X)

    

    model = Model(inputs = X_input, outputs = X, name='Two_hidden_layers')



    return model
def CNN_model(input_shape):

    """The model of simple convolutional neural network, the model is similar to NeLet5 but with a few adjustments"""

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    

    X = Conv2D(6, (5, 5), strides = (1, 1), name = 'conv1')(X)

    X = BatchNormalization(axis = 3, name = 'bn1')(X)

    X = Activation('relu')(X)

    

    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    

    X = Conv2D(16, (5, 5), strides = (1, 1), name = 'conv2')(X)

    X = BatchNormalization(axis = 3, name = 'bn2')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool2')(X)



    X = Flatten()(X)

    X = Dense(84, activation="relu")(X)

    X = Dense(10, activation="softmax")(X)

    

    model = Model(inputs = X_input, outputs = X, name='CNN')

    return model
print(X_train.shape[1:])

NN_model = two_layers(X_train.shape[1:])

NN_model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
NN_model.fit(x = X_train, y = Y_train, epochs = 30, batch_size = 32)
nn_predictions = NN_model.evaluate(x = X_test, y = Y_test)



print(nn_predictions)
X_train = X_train.reshape((60000,28,28,1))

X_test = X_test.reshape((10000,28,28,1))

print(X_train.shape)

cnn_model = CNN_model(X_train.shape[1:])

cnn_model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

cnn_model.fit(x = X_train, y = Y_train, epochs = 30, batch_size = 32)
cnn_predictions = cnn_model.evaluate(x = X_test, y = Y_test)



print(cnn_predictions)