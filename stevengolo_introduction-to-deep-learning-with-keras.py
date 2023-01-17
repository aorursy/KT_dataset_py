# Display figure in the notebook

%matplotlib inline 
# Load packages

import keras

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from keras import initializers, optimizers

from keras.layers.core import Dense, Activation

from keras.models import Sequential

from keras.utils.np_utils import to_categorical



from sklearn import preprocessing
# Define some functions

def plot_mnist(data, index, label=None):

    """Plot one image from the mnist dataset."""

    fig = plt.figure(figsize=(3, 3))

    if type(data) == pd.DataFrame:

        plt.imshow(np.asarray(data.iloc[index, 1:]).reshape((HEIGHT, WIDTH)),

                   cmap=plt.cm.gray_r,

                   interpolation='nearest')

        plt.title(f"Image label: {data.loc[index, 'label']}")

    else:

        plt.imshow(data[index].reshape((HEIGHT, WIDTH)),

                   cmap=plt.cm.gray_r,

                   interpolation='nearest')

        plt.title(f"Image label: {label}")

    

    plt.axis('off')

    return fig

    

def plot_history(history):

    """Plot the history of the training of a neural network."""

    fig = plt.figure(figsize=(10, 5))

    

    ax1 = fig.add_subplot(121)

    ax1.set(title='Model loss', xlabel='Epochs', ylabel='Loss')

    

    ax2 = fig.add_subplot(122)

    ax2.set(title='Model accuracy', xlabel='Epochs', ylabel='Accuracy')

    

    if len(history) == 2:

        ax1.plot(history['loss'])

        ax2.plot(history['acc'])

    else:

        for lr in history:

            ax1.plot(history[lr]['loss'], label=lr)

            ax2.plot(history[lr]['acc'], label=lr)

        ax1.legend(title='Learning rate')

        ax2.legend(title='Learning rate')

        

    return fig
# Load the data

digits_train = pd.read_csv('../input/train.csv')

digits_test = pd.read_csv('../input/test.csv')
# Define some global parameters

HEIGHT = 28 # Height of an image

WIDTH = 28 # Width of an image

PIXEL_NUMBER = 784 # Number of pixel in an image

PIXEL_VALUE = 255 # Maximum pixel value in an image
# Print an image

sample_index = 42



plot_mnist(digits_train, sample_index)

plt.show()
# Extract and convert the pixel as numpy array with dtype='float32'

train = np.asarray(digits_train.iloc[:, 1:], dtype='float32')

test = np.asarray(digits_test, dtype='float32')



train_target = np.asarray(digits_train.loc[:, 'label'], dtype='int32')
# Scale the data

scaler = preprocessing.StandardScaler()

train_scale = scaler.fit_transform(train)

test_scale = scaler.transform(test)



# In order to get back to the original data, you can do

# scaler.invere_transform(train.scale)
# Print a scaled image

sample_index = 42



plot_mnist(train_scale, sample_index, train_target[sample_index])

plt.show()
# Encoded the target vector as one-hot-encoding vector.

target = to_categorical(train_target)
# Define some parameters

N = train.shape[1] # Length of one data

H = 100 # Dimension of the hidden layer

K = 10 # Dimension of the output layer (number of classes to predict)

lr = 0.1 # Learning rate for the loss function

epochs = 15 # Number of epochs for the NN

batch_size = 32 # Size of the batch



# Define the model

model = Sequential()

model.add(Dense(H, input_dim=N, activation='tanh'))

model.add(Dense(K, activation='softmax'))



# Print the model

model.summary()
# Define the loss function with the optimizer

model.compile(optimizer=optimizers.SGD(lr=lr),

             loss='categorical_crossentropy',

             metrics=['accuracy'])



# Fit the model

history = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0)
# Let's plot the results

plot_history(history.history)

plt.show()
lrs = np.logspace(-3, 1, num=5)

history = dict()

for lr in lrs:

    model = Sequential()

    model.add(Dense(H, input_dim=N, activation='tanh'))

    model.add(Dense(K, activation='softmax'))

    model.compile(optimizer=optimizers.SGD(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    history[lr] = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0).history
plot_history(history)

plt.show()
model = Sequential()

model.add(Dense(H, input_dim=N, activation='tanh'))

model.add(Dense(K, activation='softmax'))

model.compile(optimizer=optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),

             loss='categorical_crossentropy',

             metrics=['accuracy'])

history = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0)
plot_history(history.history)

plt.show()
model = Sequential()

model.add(Dense(H, input_dim=N, activation='tanh'))

model.add(Dense(K, activation='softmax'))

model.compile(optimizer=optimizers.Adam(),

             loss='categorical_crossentropy',

             metrics=['accuracy'])

history = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0)
plot_history(history.history)

plt.show()
model = Sequential()

model.add(Dense(H, input_dim=N, activation='relu'))

model.add(Dense(H, activation='relu'))

model.add(Dense(K, activation='softmax'))

model.compile(optimizer=optimizers.Adam(),

             loss='categorical_crossentropy',

             metrics=['accuracy'])

history = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0)
plot_history(history.history)

plt.show()
model = Sequential()

model.add(Dense(H, input_dim=N, activation='tanh'))

model.add(Dense(K, activation='softmax'))

model.compile(optimizer=optimizers.Adadelta(),

             loss='categorical_crossentropy',

             metrics=['accuracy'])

history = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0)
plot_history(history.history)

plt.show()
prediction = model.predict_classes(test)
# Let's look at some prediction

plot_mnist(test, sample_index, prediction[sample_index])

plt.show()



plot_mnist(test, 0, prediction[0])

plt.show()
# Export the results in order to have an average accuracy

submission = pd.read_csv('../input/sample_submission.csv')

submission.loc[:, 'Label'] = prediction

submission.to_csv('submission.csv', index=False)
# Define a random normal initializers.

normal_init = initializers.RandomNormal(stddev=0.01)



model = Sequential()

model.add(Dense(H, input_dim=N, activation='tanh', kernel_initializer=normal_init))

model.add(Dense(K, activation='tanh', kernel_initializer=normal_init))

model.add(Dense(K, activation='softmax', kernel_initializer=normal_init))



model.compile(optimizer=optimizers.SGD(lr=lr),

             loss='categorical_crossentropy',

             metrics=['accuracy'])



model.summary()
w = model.layers[0].weights[0].eval(keras.backend.get_session())
print(f'Initialization weights: \n\t - mean = {np.round(w.mean(), 2)}\n\t - standard deviation = {np.round(w.std(), 2)}')
b = model.layers[0].weights[1].eval(keras.backend.get_session())
print(f'Initialization bias: \n\t - mean = {b.mean()}\n\t - standard deviation = {b.std()}')
history = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0)
plot_history(history.history)

plt.show()
# Define different initializations

init_list = [

    ('Glorot Uniform Init', 'glorot_uniform'),

    ('Small Scale Init', initializers.RandomNormal(stddev=1e-3)),

    ('Large Scale Init', initializers.RandomNormal(stddev=1)),

    ('Zero Weights Init', 'zero')

]



optimizer_list = [

    ('SGD', optimizers.SGD(lr=lr)),

    ('Adam', optimizers.Adam()),

    ('SGD + Nesterov momentum', optimizers.SGD(lr=lr, momentum=0.9, nesterov=True))

]



history = dict()

for optimizer_name, optimizer in optimizer_list:

    for init_name, init in init_list:

        model = Sequential()

        model.add(Dense(H, input_dim=N, activation='tanh', kernel_initializer=init))

        model.add(Dense(K, activation='tanh', kernel_initializer=init))

        model.add(Dense(K, activation='softmax', kernel_initializer=init))

        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        history[(optimizer_name, init_name)] = model.fit(train_scale, target, epochs=epochs, batch_size=batch_size, verbose=0)
for optimizer_name, init_name in history:

    plot_history(history[(optimizer_name, init_name)].history)

    plt.suptitle(f"Optimizer: {optimizer_name}, Initialization: {init_name}")

    plt.show()