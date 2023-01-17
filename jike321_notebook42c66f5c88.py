# Load the training set and transform it into the shape we need.



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tflearn.data_utils as du



def load_traindata():

    data = pd.read_csv('../input/train.csv')

    label = np.array(data['label'])

    pixels = data[['pixel%d' % i for i in range(784)]]

    pixels2 = []

    for _, s in pixels.iterrows():

        pixels2.append(np.array(s).reshape([28, 28]))

    pixels, _ = du.featurewise_zero_center(np.array(pixels2))

    return onehot(label, 10), pixels.reshape((-1, 28, 28, 1))



def onehot(series, n_categories):

    """

    One-hot encoding for labels

    """

    return np.array([[1 if i == j else 0 for j in range(n_categories)] for i in series])



labels, pixels = load_traindata()





# Any results you write to the current directory are saved as output.
import tensorflow as tf

import tflearn



def build_net():

    net = tflearn.input_data((None, 28, 28, 1), name='input')

    net = tflearn.layers.conv_2d(net, 32, 3, activation='relu', regularizer='L2')

    net = tflearn.layers.max_pool_2d(net, 3)

    net = tflearn.layers.conv_2d(net, 64, 3, activation='relu', regularizer='L2')

    net = tflearn.layers.max_pool_2d(net, 3)

    net = tflearn.local_response_normalization(net)

    net = tflearn.fully_connected(net, 128, activation='relu')

    net = tflearn.dropout(net, 0.8)

    net = tflearn.fully_connected(net, 256, activation='relu')

    net = tflearn.dropout(net, 0.8)

    net = tflearn.fully_connected(net, 10, name='output')

    net = tflearn.regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='regression')

    model = tflearn.DNN(net, tensorboard_verbose=0)

    return model
labels, pixels = load_traindata()

net = build_net()

net.fit(pixels, labels, n_epoch=10)
def load_testset():

    data = pd.read_csv('../input/test.csv')

    pixels = []

    for _, s in data.iterrows():

        pixels.append(np.array(s).reshape([28, 28]))

    pixels, _ = du.featurewise_zero_center(np.array(pixels))

    return pixels.reshape((-1, 28, 28, 1))
testset = load_testset()

pred = net.predict(testset)

pred = [x[1] for x in pred]

s = pd.DataFrame({'ImageId': list(range(len(pred))), 'Label':pred})

s.to_csv('output.csv', index=False)