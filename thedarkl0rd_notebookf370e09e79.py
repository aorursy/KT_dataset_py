# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import math

import numpy as np

import tensorflow as tf

from matplotlib import pyplot as plt



from keras.models import Sequential

from keras.layers import Dense, Activation, Reshape, Flatten, Input

from keras.layers import Conv1D, MaxPooling1D, Convolution1D, GlobalAveragePooling1D

from keras.layers import BatchNormalization, Dropout

from keras.layers import LSTM, GRU, Masking

from keras.optimizers import SGD, RMSprop, Adam
def train_test_split(X, Y, train_fraction=0.8, shuffle=False):

    N = len(Y)

    if shuffle == True:

        from random import shuffle

        idx = shuffle(np.arange(N))

        X = X[idx]

        Y = Y[idx]

    

    x_train = X[:math.ceil(N*train_fraction)]

    y_train = Y[:math.ceil(N*train_fraction)]

    x_test = X[math.ceil(N*train_fraction):]

    y_test = Y[math.ceil(N*train_fraction):]

    

    return x_train, y_train, x_test, y_test
def read_clean_dataset(summary=False):

    """Load clean dataset."""

    data = np.load('../input/full.npz')

    features = data['features']

    labels = data['labels']

    if summary:

        print('Loaded clean data:')

        print('Data has shape = {}, contains {} unique labels'.format(

            features.shape, len(np.unique(labels))))

        fig = plt.figure(figsize=(4,4))

        ax = fig.add_subplot(111)

        im = ax.imshow(features, aspect='auto')

        ax.set_xlabel('feature dim')

        ax.set_ylabel(r'$x_{i}$')

        ax.set_title('Clean Features')

        cbar = fig.colorbar(im)

        plt.show()

    return features, labels
def one_hot(y):

    """

    Convert dataset labels to one-hot.

    Method will not work for datasets with non-integer/index labels.

    """

    num_class = len(np.unique(y))

    y_one_hot = np.zeros((len(y), num_class), dtype=np.int32)

    for i, y_i in zip(range(len(y)), y):

        y_one_hot[i,y_i] = 1

    return y_one_hot

# Load data

x, y = read_clean_dataset(summary=True)

y = one_hot(y)

x_train, y_train, x_test, y_test = train_test_split(x, y)

feat_dim = x.shape[1]

out_dim = y.shape[1]

#RNN train and test set

x_train_rnn = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))

x_test_rnn = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

#CNN train and test set

x_train_cnn = x_train.reshape((x_train.shape[0], x_train.shape[1],1))

x_test_cnn = x_test.reshape((x_test.shape[0], x_test.shape[1],1))
def categorical_crossentropy(y_true, y_pred):

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, 

                                                   logits=y_pred)

    return loss



# Using only basic tensorflow ops, implement the following functions

# TODO: cosine distance

def cosine_distance(y_true, y_pred):

    """

    Cosine distance loss using only basic tensorflow ops.

    (Do not use tf.losses, tf.nn, etc.)

    """

    y_true = tf.nn.l2_normalize(y_true, dim=-1)

    y_pred = tf.nn.l2_normalize(y_pred, dim=-1)

    return -tf.reduce_sum(y_true * y_pred, axis = -1, keepdims = False)

    



# TODO: regression error

# mean absolute error ^ n

def regression_error(y_true, y_pred, p=2):

    """

    Regression error loss using only basic tensorflow ops.

    (Do not use tf.losses, tf.nn, etc.)

    """

    return tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1,keepdims = False)



# TODO: hinge loss

def hinge_loss(y_true, y_pred):

    """

    Hinge loss loss using only basic tensorflow ops.

    (Do not use tf.losses, tf.nn, etc.)

    """

    

    return tf.reduce_mean(tf.maximum(1. - y_true * y_pred, 0.), axis=-1, keepdims = False)
# Network architectures

def mlp(model, train, num_nodes=64, num_layers=3, 

              activation='relu', ouput_activation='softmax'):

    """

    Builds a basic neural network in Keras.

    By default, assumes a multiclass classification 

    problem (softmax output).

    """

    model.add(Dense(num_nodes, activation=activation, 

                    input_shape=(feat_dim,)))

    for l in range(num_layers-1):

        model.add(Dense(num_nodes, activation=activation))

    model.add(Dense(out_dim, activation='softmax'))

    return model



# TODO: function to build a CNN

def cnn(model,train_cnn,num_nodes = 64,activation = 'relu'):

    #print(train_cnn.shape)

    model.add(Convolution1D(nb_filter=num_nodes, filter_length=1, input_shape=(train_cnn.shape[1], 1)))

    model.add(Activation(activation))

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(num_nodes, activation=activation))

    #model.add(Dense(64, activation='relu'))

    model.add(Dense(out_dim))

    model.add(Activation('softmax'))

    return model



# TODO: function to build an RNN

def rnn(model,train_rnn, num_nodes=128,ouput_activation='softmax',**kwargs):

    """

    Build an RNN to handle 1D sequences.

    """

    model.add(LSTM(num_nodes, batch_input_shape=(batch_size, train_rnn.shape[1], train_rnn.shape[2]), 

                     stateful=True))

    #model.add(LSTM(16, return_sequences=True))

    model.add(Dense(out_dim, activation='softmax'))

    

    return model
def get_model(mdl,loss,save_png='plot.png'):

    #print(mdl)

    model = Sequential()

    model = mdl[0](model,mdl[1],num_nodes = 64)#build_mlp(model, num_nodes, num_layers, activation)

    # Define optimizer and compile model

    rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(loss=loss,#categorical_crossentropy, # <-- custom loss functions

                  optimizer=rmsprop,

                  metrics=['accuracy'])

    # Train model

    history = model.fit(mdl[1], y_train, 

                        batch_size=batch_size, 

                        epochs=epochs, 

                        shuffle=True, 

                        validation_data=(mdl[2], y_test))

    # Plot training history: loss as a function of epoch

    fig = plt.figure(figsize=(10,4))

    ax = fig.add_subplot(121)

    for k in ['loss', 'val_loss']:

        ax.plot(history.history[k], label=k)

    ax.set_xlabel('epoch')

    ax.set_ylabel('loss')

    plt.legend()

    # Plot training history: accuracy as a function of epoch

    ax = fig.add_subplot(122)

    for k in ['acc', 'val_acc']:

        ax.plot(history.history[k], label=k)

    ax.set_xlabel('epoch')

    ax.set_ylabel('accuracy')

    plt.legend()

    fig.savefig(save_png)

    plt.show()

    

    return model
# Using an MLP to perform classification

num_nodes = 256

num_layers = 2

activation = 'relu'

batch_size = 500

epochs = 10



#x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))

#x_test = x_test.reshape((x_test.shape[0], x_test.shape[1],1))

#print(x_train_rnn.shape)

# Create model

models ={

    'mlp':(mlp,x_train,x_test),

    'cnn':(cnn,x_train_cnn,x_test_cnn),

    'rnn':(rnn,x_train_rnn,x_test_rnn)

        }

losses = {

    'cc':categorical_crossentropy,

    'hl':hinge_loss, 

    'cd':cosine_distance, 

    're':regression_error

    }

#models['cnn'][2].shape

for mod in models:

    for loss in losses:

        get_model(models[mod],losses[loss],mod+'-'+loss+'.png')

#final_model = get_model(models['cnn'],losses['hl'],'cnn-hl.png')
def read_corrupted_dataset(summary=False):

    """Load corrupted dataset."""

    data = np.load('../input/corrupted_series.npz')

    features = data['features']

    length = data['length']

    if summary:

        print('Loaded corrupted data:')

        print('Data has shape = {}, average sequence length = {:0.2f}'.format(

            features.shape, np.mean(length)))

        fig = plt.figure(figsize=(8,4))

        ax = fig.add_subplot(121)

        im = ax.imshow(features, aspect='auto')

        ax.set_xlabel('feature dim')

        ax.set_ylabel(r'$x_{i}$')

        ax.set_title('Corrupted Features')

        ax = fig.add_subplot(122)

        ax.hist(length, bins=20)

        ax.set_title('Histogram of Sequence Length')

        ax.set_xlabel('length')

        ax.set_ylabel('count')

        plt.show()

    return features, length
x, x_len = read_corrupted_dataset(summary=True)

final_model = get_model(models['cnn'],losses['cd']) 

test_labels = final_model.predict(x)

#converting back from one hot representation

print()

test_labels = np.argmax(test_labels, axis=1)#[ np.where(r==1)[0][0] for r in test_labels ]

np.savez('corrupt_labels', test_labels)

print(test_labels.shape)
# TODO: function to build an RNN

def rnn_regressor(model,train_rnn, num_nodes=128,ouput_activation='softmax',**kwargs):

    model.add(LSTM(num_nodes, input_shape=( train_rnn.shape[1], train_rnn.shape[2]), stateful=False))

    #model.add(LSTM(16, return_sequences=False))

    #model.add(Dense(30, activation='relu'))

    model.add(Dense(1, activation='linear'))

    return model

def cnn_regressor(model,train_cnn,num_nodes = 64,activation = 'relu'):

    #print(train_cnn.shape)

    model.add(Convolution1D(nb_filter=num_nodes, filter_length=1, input_shape=(train_cnn.shape[1], 1)))

    model.add(Activation(activation))

    model.add(Flatten())

    model.add(Dropout(0.4))

    model.add(Dense(num_nodes, activation=activation))

    #model.add(Dense(64, activation='relu'))

    model.add(Dense(1))

    model.add(Activation('linear'))

    return model

batch_size = 16

epochs = 10

results = []

index = 0

train = x[index][:x_len[index]-1]

y = x[0][1:x_len[index]]



model = Sequential()

model = cnn_regressor(model,train.reshape((train.shape[0], 1, 1)),num_nodes = 64)#build_mlp(model, num_nodes, num_layers, activation)

# Define optimizer and compile model

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-03, decay=0.0)

model.compile(loss=regression_error, # <-- custom loss functions

              optimizer=rmsprop,

              metrics=['accuracy'])

# Train model

history = model.fit(train.reshape((train.shape[0], 1,1)), y, 

                    #batch_size=batch_size, 

                    epochs=epochs, 

                    shuffle=False)

for index in range(30000):

    iteration = 25

    train.reshape((train.shape[0], 1, 1))[0]

    temp = np.zeros(25)

    temp[0] = x[index][x_len[index]-1]

    #print(temp)

    i = 0

    while(iteration>0):

        pred = model.predict(temp.reshape((temp.shape[0], 1, 1)))

        temp[i] = pred[i-1]

        i = i+1

        iteration = iteration -1

    results.append(temp)

    if index%1000 == 0:

        print(index)

print(len(results))



np.savez('corrupt_prediction.npz', np.array(results))