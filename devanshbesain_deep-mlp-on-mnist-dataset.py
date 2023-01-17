import pandas as pd

import tflearn

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
tf.reset_default_graph()
mnist = input_data.read_data_sets('../input/', one_hot=True)
type(mnist)


X = mnist.train.images

Y =  mnist.train.labels

testX =  mnist.test.images

testY = mnist.test.labels

valX = mnist.validation.images

valY = mnist.validation.labels

del mnist
input_layer = tflearn.input_data(shape=[None, 784])

dense1 = tflearn.fully_connected(input_layer, 64, activation='relu',

                                 regularizer='L2', weight_decay=0.001)

dropout1 = tflearn.dropout(dense1, 0.7)

dense2 = tflearn.fully_connected(dropout1, 128, activation='relu',

                                 regularizer='L2', weight_decay=0.001)

dropout2 = tflearn.dropout(dense2, 0.7)

dense3 = tflearn.fully_connected(dropout2, 128, activation='relu',

                                 regularizer='L2', weight_decay=0.001)

dropout3 = tflearn.dropout(dense2, 0.7)

softmax = tflearn.fully_connected(dropout3, 10, activation='softmax')



sgd = tflearn.Adam(learning_rate=0.01)

top_k = tflearn.metrics.Top_k(5)

net = tflearn.regression(softmax, optimizer=sgd, metric=top_k,

loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=3)

model.fit(X, Y, n_epoch=2, validation_set=(testX, testY),show_metric=True, run_id="dense_model" )
del X

del Y

del testX

del testY
import matplotlib.pyplot as plt

%matplotlib inline
plt.imshow(valX[20].reshape(28,28))
pd.DataFrame(model.predict(valX[20].reshape(1,784)))
model.predict_label(valX[20].reshape(1,784))
y = model.predict(valX)
# Sanity check

y.shape == valY.shape
model.evaluate(valX,valY)