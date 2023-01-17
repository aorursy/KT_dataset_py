# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import matplotlib.pyplot as plot

import tensorflow as tf

import random

import torch

import torchvision

from sklearn.model_selection import * 

from catboost import *

from keras import *

from keras.layers.convolutional import Conv2D

from keras.layers import *

from tensorflow.nn import *

from keras.callbacks import *

from keras.models import *

from keras.optimizers import *

from keras.preprocessing import image

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
random.seed(2)

np.random.seed(2)

random_seed = 2
train = pd.read_csv('../input/mnist_train.csv')

test = pd.read_csv('../input/mnist_test.csv')
X_train = train.drop('label', axis = 1)

y_train = train.label.to_frame()



X_test = test.drop('label', axis = 1)

y_test = test.label.to_frame()
X_train.head()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .3, random_state = 2)
cb = CatBoostClassifier(task_type = "GPU", eval_metric = "Accuracy", verbose = 0, od_wait = 120, od_type = 'Iter')

cb.fit(X_train, y_train, eval_set = (X_val, y_val), use_best_model = True, plot = True)
train = pd.read_csv('../input/mnist_train.csv')

test = pd.read_csv('../input/mnist_test.csv')



X_train = train.drop('label', axis = 1)

y_train = train.label.to_frame()



X_test = test.drop('label', axis = 1)

y_test = test.label.to_frame()
X_train = X_train.values.reshape(X_train.shape[0], 28, 28, 1)

X_test = X_test.values.reshape(X_test.shape[0], 28, 28, 1)



X_train = X_train.astype(float)

X_test = X_test.astype(float)



X_train /= 255

X_test /= 255
model = Sequential()

model.add(Convolution2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu', input_shape=(28, 28, 1)))

# model.add(Conv2D(28, kernel_size = (3, 3), input_shape = (28, 28, 1)))

# model.add(Permute((2, 2), input_shape=(28, 28)))

model.add(MaxPooling2D(pool_size = (2, 2), input_shape=(28, 28)))

model.add(Flatten(input_shape = (28, 28)))

model.add(Dense(128, activation = relu))

# model.add(RepeatVector(3))

model.add(Dense(512, activation = relu)) 

model.add(Dropout(rate = 0.25))

model.add(Dense(20, activation = softmax)) 

# model.add(Dense(25, activation = softmax)) 

# model.add(Dropout(rate = 0.5))



model2 = Sequential()

model2.add(Convolution2D(filters = 64, kernel_size = 2, padding = 'same', activation = 'relu', input_shape=(28, 28, 1)))

model2.add(MaxPooling2D(pool_size = (2, 2), input_shape=(28, 28)))

model2.add(Flatten(input_shape = (28, 28)))

model2.add(Dense(128, activation = relu))

model2.add(Dense(512, activation = relu)) 

model2.add(Dropout(rate = 0.25))

model2.add(Dense(20, activation = softmax)) 
NoNaN = TerminateOnNaN()

earlyStopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 0, mode = 'auto')

mcp_save = ModelCheckpoint('model.best.hdf5', save_best_only = True, monitor = 'val_loss', mode = 'auto')

lr_loss = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 7, verbose = 1, mode = 'auto')

rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = 0.5, min_lr = 0.00001)
generator = image.ImageDataGenerator()



model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])



batches = generator.flow(X_train, y_train, batch_size = 64)

val_batches = generator.flow(X_test, y_test, batch_size = 64)



model.fit_generator(batches, batches.n, nb_epoch = 3, validation_data = val_batches, nb_val_samples = val_batches.n, verbose = 1, callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN, rate_reduction])
model2.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model2.fit(X_train, y_train, batch_size = 86, epochs = 50, callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN, rate_reduction], validation_data = (X_test, y_test))                    # Validation data only for tests



model2.load_weights('model.best.hdf5')                                                                                                                                                   # Loads best weights



model2.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

model2.fit(X_train, y_train, batch_size = 86, epochs = 30, callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN, rate_reduction], validation_split = .3)  
print("Точность первой модели:", round(model.evaluate(X_test, y_test, verbose = 0)[1] * 100), "%", "(", model.evaluate(X_test, y_test, verbose = 0)[1], ")") 

print("Точность второй модели:", round(model2.evaluate(X_test, y_test, verbose = 0)[1] * 100), "%", "(", model2.evaluate(X_test, y_test, verbose = 0)[1], ")") 
test_id = 0



plot.imshow(X_test[test_id].reshape(28, 28), cmap = 'Greys')

pred = model.predict(X_test[test_id].reshape(1, 28, 28, 1), callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN])

print(pred.argmax())
test_id = 0

preds = list()



for test_id in range(len(X_test)):

    pred = model.predict(X_test[test_id].reshape(1, 28, 28, 1), callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN, rate_reduction])

    preds.append(pred.argmax())



keras_pred = pd.DataFrame({'label': preds})



keras_pred.head(3)



test_id2 = 0

preds2 = list()



for test_id2 in range(len(X_test)):

    pred2 = model2.predict(X_test[test_id2].reshape(1, 28, 28, 1), callbacks = [earlyStopping, mcp_save, lr_loss, NoNaN, rate_reduction])

    preds2.append(pred2.argmax())



keras_pred2 = pd.DataFrame({'label': preds2})



keras_pred2.head(3)
train = pd.read_csv('../input/mnist_train.csv')

test = pd.read_csv('../input/mnist_test.csv')



X_train = train.drop('label', axis = 1)

y_train = train.label.to_frame()



X_test = test.drop('label', axis = 1)

y_test = test.label.to_frame()
cb_predictions = cb.predict(X_test)

print("Accuracy: {}%".format(accuracy_score(cb_predictions, y_test) * 100))

round(accuracy_score(y_test, cb_predictions) * 100)
train = pd.read_csv('../input/mnist_train.csv')

test = pd.read_csv('../input/mnist_test.csv')



X_train = train.drop('label', axis = 1)

y_train = train.label.to_frame()



X_test = test.drop('label', axis = 1)

y_test = test.label.to_frame()
from sklearn.neural_network import MLPClassifier



mlp_classifier = MLPClassifier(verbose = True, activation = 'relu', batch_size = 86, learning_rate = 'adaptive', max_iter = 70, random_state = 2, early_stopping = True, momentum = 0.5, n_iter_no_change = 12)

mlp_classifier = mlp_classifier.fit(X_train, y_train)
test_id = 1



mlp_classifier.predict(X_test.values[test_id, :].reshape(1, 784))

mlp_predictions = mlp_classifier.predict(X_test)

print("Accuracy: {}%".format(accuracy_score(y_test, mlp_predictions) * 100))
n_neurons = 128

n_epochs = 20

n_steps = 28 

learning_rate = 0.0005

batch_size = 128

n_inputs = 28 

n_outputs = 10 



Data = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

test = tf.placeholder(tf.int32, [None])

 

cell = tf.nn.rnn_cell.BasicRNNCell(num_units = n_neurons)

output, state = tf.nn.dynamic_rnn(cell, Data, dtype = tf.float32)

logits = tf.layers.dense(state, n_outputs)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = test, logits = logits)

loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

prediction = tf.nn.in_top_k(logits, test, 1)

accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))



from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/")

X_test = mnist.test.images

X_test = X_test.reshape([-1, n_steps, n_inputs])

y_test = mnist.test.labels



print("Training...")

with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    n_batches = mnist.train.num_examples // batch_size

    for epoch in range(n_epochs):

        for batch in range(n_batches):

            X_train, y_train = mnist.train.next_batch(batch_size)

            X_train = X_train.reshape([-1, n_steps, n_inputs])

            session.run(optimizer, feed_dict = {Data: X_train, test: y_train})

        loss_train, acc_train = session.run([loss, accuracy], feed_dict = {Data: X_train, test: y_train})

        print('Epoch: {}, Train Loss: {:.3f}, Train Acc: {:.3f}'.format(epoch + 1, loss_train, acc_train))

    loss_test, acc_test = session.run([loss, accuracy], feed_dict = {Data: X_test, test: y_test})

    print('Test Loss: {:.3f}, Test Accuracy: {:.3f}%'.format(loss_test, round(acc_test * 100)))

     

session.close()