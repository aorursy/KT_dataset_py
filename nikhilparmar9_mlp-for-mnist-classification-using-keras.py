# if you keras is not using tensorflow as backend set "KERAS_BACKEND=tensorflow" use this command

from keras.utils import np_utils 

from keras.datasets import mnist 

import seaborn as sns

from keras.initializers import RandomNormal

from keras.layers.normalization import BatchNormalization

print("DONE")
%matplotlib notebook

import matplotlib.pyplot as plt

import numpy as np

import time

# https://gist.github.com/greydanus/f6eee59eaf1d90fcb3b534a25362cea4

# https://stackoverflow.com/a/14434334

# this function is used to update the plots for each epoch and error

def plt_dynamic(fig,x, vy, ty, ax, colors=['b']):

    

    ax.plot(x, vy, 'b', label="Validation Loss")

    ax.plot(x, ty, 'r', label="Train Loss")

    plt.legend()

    plt.grid()

    fig.canvas.draw()
def printLossPlot(modelName):

    score = modelName.evaluate(X_test, Y_test, verbose=0) 

    print('Test score:', score[0]) 

    print('Test accuracy:', score[1])

#     fig = plt.figure()

    fig,ax = plt.subplots(1,1)

    ax.set_xlabel('epoch') ; ax.set_ylabel('Categorical Crossentropy Loss')



    # list of epoch numbers

    x = list(range(1,nb_epoch+1))

    vy = history.history['val_loss']

    ty = history.history['loss']

    plt_dynamic(fig,x, vy, ty, ax)
# the data, shuffled and split between train and test sets 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d, %d)"%(X_train.shape[1], X_train.shape[2]))

print("Number of training examples :", X_test.shape[0], "and each image is of shape (%d, %d)"%(X_test.shape[1], X_test.shape[2]))
X_train.shape[1]*X_train.shape[2]
# if you observe the input shape its 2 dimensional vector

# for each image we have a (28*28) vector

# we will convert the (28*28) vector into single dimensional vector of 1 * 784 



X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]) 

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2]) 
# after converting the input images from 3d to 2d vectors



print("Number of training examples :", X_train.shape[0], "and each image is of shape (%d)"%(X_train.shape[1]))

print("Number of training examples :", X_test.shape[0], "and each image is of shape (%d)"%(X_test.shape[1]))
# An example data point

print(X_train[0])
# if we observe the above matrix each cell is having a value between 0-255

# before we move to apply machine learning algorithms lets try to normalize the data

# X => (X - Xmin)/(Xmax-Xmin) = X/255



X_train = X_train/255

X_test = X_test/255
# example data point after normlizing

print(X_train[0])
# here we are having a class number for each image

print("Class label of first image :", y_train[0])



# lets convert this into a 10 dimensional vector

# ex: consider an image is 5 convert it into 5 => [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

# this conversion needed for MLPs 



Y_train = np_utils.to_categorical(y_train, 10) 

Y_test = np_utils.to_categorical(y_test, 10)



print("After converting the output into a vector : ",Y_train[0])
from keras.models import Sequential 

from keras.layers import Dense, Activation 
# some model parameters

output_dim = 10

input_dim = X_train.shape[1]

batch_size = 128 

nb_epoch = 20
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout

from keras.layers.normalization import BatchNormalization



model_2_d_bn_relu_rn_adam = Sequential()



model_2_d_bn_relu_rn_adam.add(Dense(784, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))

model_2_d_bn_relu_rn_adam.add(BatchNormalization())

model_2_d_bn_relu_rn_adam.add(Dropout(0.5))



model_2_d_bn_relu_rn_adam.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_2_d_bn_relu_rn_adam.add(BatchNormalization())

model_2_d_bn_relu_rn_adam.add(Dropout(0.5))



model_2_d_bn_relu_rn_adam.add(Dense(output_dim, activation='softmax'))



model_2_d_bn_relu_rn_adam.summary()



model_2_d_bn_relu_rn_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_2_d_bn_relu_rn_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_2_d_bn_relu_rn_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

from keras.initializers import he_uniform

from keras.layers.normalization import BatchNormalization

from keras.layers import Dropout



model_2_d_bn_relu_heUni_adam = Sequential()



model_2_d_bn_relu_heUni_adam.add(Dense(512, activation='relu', input_shape=(input_dim,), kernel_initializer=he_uniform(seed=None)))

model_2_d_bn_relu_heUni_adam.add(BatchNormalization())

model_2_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_2_d_bn_relu_heUni_adam.add(Dense(256, activation='relu', kernel_initializer=he_uniform(seed=None)) )

model_2_d_bn_relu_heUni_adam.add(BatchNormalization())

model_2_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_2_d_bn_relu_heUni_adam.add(Dense(output_dim, activation='softmax'))





model_2_d_bn_relu_heUni_adam.summary()



model_2_d_bn_relu_heUni_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_2_d_bn_relu_heUni_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_2_d_bn_relu_heUni_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras

from keras.initializers import he_normal

from keras.layers.normalization import BatchNormalization

from keras.layers import Dropout



model_2_d_bn_relu_heNor_adam = Sequential()



model_2_d_bn_relu_heNor_adam.add(Dense(600, activation='relu', input_shape=(input_dim,), kernel_initializer=he_normal(seed=None)))

model_2_d_bn_relu_heNor_adam.add(BatchNormalization())

model_2_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_2_d_bn_relu_heNor_adam.add(Dense(256, activation='relu', kernel_initializer=he_normal(seed=None)) )

model_2_d_bn_relu_heNor_adam.add(BatchNormalization())

model_2_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_2_d_bn_relu_heNor_adam.add(Dense(output_dim, activation='softmax'))





model_2_d_bn_relu_heNor_adam.summary()



model_2_d_bn_relu_heNor_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_2_d_bn_relu_heNor_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_2_d_bn_relu_heNor_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout



model_3_d_bn_relu_rn_adam = Sequential()



model_3_d_bn_relu_rn_adam.add(Dense(784, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))

model_3_d_bn_relu_rn_adam.add(BatchNormalization())

model_3_d_bn_relu_rn_adam.add(Dropout(0.5))



model_3_d_bn_relu_rn_adam.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_3_d_bn_relu_rn_adam.add(BatchNormalization())

model_3_d_bn_relu_rn_adam.add(Dropout(0.5))



model_3_d_bn_relu_rn_adam.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_3_d_bn_relu_rn_adam.add(BatchNormalization())

model_3_d_bn_relu_rn_adam.add(Dropout(0.5))



model_3_d_bn_relu_rn_adam.add(Dense(output_dim, activation='softmax'))





model_3_d_bn_relu_rn_adam.summary()



model_3_d_bn_relu_rn_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_3_d_bn_relu_rn_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_3_d_bn_relu_rn_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout



model_3_d_bn_relu_heUni_adam = Sequential(name="MLP")



model_3_d_bn_relu_heUni_adam.add(Dense(500, activation='relu', input_shape=(input_dim,), kernel_initializer=he_uniform(seed=None)))

model_3_d_bn_relu_heUni_adam.add(BatchNormalization())

model_3_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_3_d_bn_relu_heUni_adam.add(Dense(256, activation='relu', kernel_initializer=he_uniform(seed=None)) )

model_3_d_bn_relu_heUni_adam.add(BatchNormalization())

model_3_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_3_d_bn_relu_heUni_adam.add(Dense(128, activation='relu', kernel_initializer=he_uniform(seed=None)) )

model_3_d_bn_relu_heUni_adam.add(BatchNormalization())

model_3_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_3_d_bn_relu_heUni_adam.add(Dense(output_dim, activation='softmax'))



for i, layer in enumerate(model_3_d_bn_relu_heUni_adam.layers):

    layer.name = 'layer_' + str(i)

model_3_d_bn_relu_heUni_adam.summary()



model_3_d_bn_relu_heUni_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_3_d_bn_relu_heUni_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_3_d_bn_relu_heUni_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout



model_3_d_bn_relu_heNor_adam = Sequential(name="MLP")



model_3_d_bn_relu_heNor_adam.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_initializer=he_normal(seed=None)))

model_3_d_bn_relu_heNor_adam.add(BatchNormalization())

model_3_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_3_d_bn_relu_heNor_adam.add(Dense(64, activation='relu', kernel_initializer=he_normal(seed=None)) )

model_3_d_bn_relu_heNor_adam.add(BatchNormalization())

model_3_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_3_d_bn_relu_heNor_adam.add(Dense(32, activation='relu', kernel_initializer=he_normal(seed=None)) )

model_3_d_bn_relu_heNor_adam.add(BatchNormalization())

model_3_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_3_d_bn_relu_heNor_adam.add(Dense(output_dim, activation='softmax'))



for i, layer in enumerate(model_3_d_bn_relu_heNor_adam.layers):

    layer.name = 'layer_' + str(i)

model_3_d_bn_relu_heNor_adam.summary()



model_3_d_bn_relu_heNor_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



history = model_3_d_bn_relu_heNor_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_3_d_bn_relu_heNor_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout



model_5_d_bn_relu_rn_adam = Sequential()



model_5_d_bn_relu_rn_adam.add(Dense(784, activation='relu', input_shape=(input_dim,), kernel_initializer=RandomNormal(mean=0.0, stddev=0.039, seed=None)))

model_5_d_bn_relu_rn_adam.add(BatchNormalization())

model_5_d_bn_relu_rn_adam.add(Dropout(0.5))



model_5_d_bn_relu_rn_adam.add(Dense(512, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_5_d_bn_relu_rn_adam.add(BatchNormalization())

model_5_d_bn_relu_rn_adam.add(Dropout(0.5))



model_5_d_bn_relu_rn_adam.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_5_d_bn_relu_rn_adam.add(BatchNormalization())

model_5_d_bn_relu_rn_adam.add(Dropout(0.5))



model_5_d_bn_relu_rn_adam.add(Dense(128, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_5_d_bn_relu_rn_adam.add(BatchNormalization())

model_5_d_bn_relu_rn_adam.add(Dropout(0.5))



model_5_d_bn_relu_rn_adam.add(Dense(64, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.55, seed=None)) )

model_5_d_bn_relu_rn_adam.add(BatchNormalization())

model_5_d_bn_relu_rn_adam.add(Dropout(0.5))



model_5_d_bn_relu_rn_adam.add(Dense(output_dim, activation='softmax'))





model_5_d_bn_relu_rn_adam.summary()



model_5_d_bn_relu_rn_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_d_bn_relu_rn_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_5_d_bn_relu_rn_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout



model_5_d_bn_relu_heUni_adam = Sequential()



model_5_d_bn_relu_heUni_adam.add(Dense(784, activation='relu', input_shape=(input_dim,), kernel_initializer=he_uniform(seed=None)))

model_5_d_bn_relu_heUni_adam.add(BatchNormalization())

model_5_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_5_d_bn_relu_heUni_adam.add(Dense(600, activation='relu', kernel_initializer=he_uniform(seed=None)) )

model_5_d_bn_relu_heUni_adam.add(BatchNormalization())

model_5_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_5_d_bn_relu_heUni_adam.add(Dense(512, activation='relu', kernel_initializer=he_uniform(seed=None)) )

model_5_d_bn_relu_heUni_adam.add(BatchNormalization())

model_5_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_5_d_bn_relu_heUni_adam.add(Dense(128, activation='relu', kernel_initializer=he_uniform(seed=None)) )

model_5_d_bn_relu_heUni_adam.add(BatchNormalization())

model_5_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_5_d_bn_relu_heUni_adam.add(Dense(64, activation='relu', kernel_initializer=he_uniform(seed=None)) )

model_5_d_bn_relu_heUni_adam.add(BatchNormalization())

model_5_d_bn_relu_heUni_adam.add(Dropout(0.5))



model_5_d_bn_relu_heUni_adam.add(Dense(output_dim, activation='softmax'))





model_5_d_bn_relu_heUni_adam.summary()



model_5_d_bn_relu_heUni_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_d_bn_relu_heUni_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_5_d_bn_relu_heUni_adam)
# https://stackoverflow.com/questions/34716454/where-do-i-call-the-batchnormalization-function-in-keras



from keras.layers import Dropout



model_5_d_bn_relu_heNor_adam = Sequential()



model_5_d_bn_relu_heNor_adam.add(Dense(600, activation='relu', input_shape=(input_dim,), kernel_initializer=he_normal(seed=None)))

model_5_d_bn_relu_heNor_adam.add(BatchNormalization())

model_5_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_5_d_bn_relu_heNor_adam.add(Dense(512, activation='relu', kernel_initializer=he_normal(seed=None)) )

model_5_d_bn_relu_heNor_adam.add(BatchNormalization())

model_5_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_5_d_bn_relu_heNor_adam.add(Dense(256, activation='relu', kernel_initializer=he_normal(seed=None)) )

model_5_d_bn_relu_heNor_adam.add(BatchNormalization())

model_5_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_5_d_bn_relu_heNor_adam.add(Dense(128, activation='relu', kernel_initializer=he_normal(seed=None)) )

model_5_d_bn_relu_heNor_adam.add(BatchNormalization())

model_5_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_5_d_bn_relu_heNor_adam.add(Dense(64, activation='relu', kernel_initializer=he_normal(seed=None)) )

model_5_d_bn_relu_heNor_adam.add(BatchNormalization())

model_5_d_bn_relu_heNor_adam.add(Dropout(0.5))



model_5_d_bn_relu_heNor_adam.add(Dense(output_dim, activation='softmax'))





model_5_d_bn_relu_heNor_adam.summary()



model_5_d_bn_relu_heNor_adam.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model_5_d_bn_relu_heNor_adam.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
printLossPlot(model_5_d_bn_relu_heNor_adam)