import numpy as np

import pandas as pd  # we usually use this but this time around, the data set is special

import matplotlib.pyplot as plt

from sklearn import linear_model, neural_network, decomposition



import gzip, pickle, sys

f = gzip.open('../input/mnist.pkl.gz', 'rb')

(X_train, y_train), (X_test, y_test), _ = pickle.load(f, encoding='bytes')



print(X_train.shape, y_train.shape)

img_len = int(np.sqrt(X_train.shape[1]))

print('image dimension = ' + str(img_len) + 'x' + str(img_len))
# visualize one number

def plot_num(X):

    X = X.reshape(1,-1)

    assert(X.shape[1] == img_len*img_len)

    image = np.reshape(X, (img_len, img_len) )

    _ = plt.figure()

    _ = plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')

    

# visualize 100 numbers, store numbers in matrix format where each row represents a number

def plot_num_100(X):

    assert(X.shape[1] == img_len*img_len)

    n = len(X) if (len(X) < 100) else 100

    X = X[np.random.choice(np.arange(len(X)), size=n, replace=False),:]

    

    fig = plt.figure()

    for i in range(len(X)):

        image = np.reshape(X[i,:], (img_len, img_len) )

        _ = plt.subplot(10, 10, i + 1)

        _ = plt.axis('off')

        _ = plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
# take a look at the data set to see what we are given

print(X_train.shape)

print(X_train.min(), X_train.max())

#print(X_train[0,:])



print(y_train.shape)

print(y_train[0])



# plot out an examples to see what we are looking at

plot_num(X_train[0,:]);
# visualize 100 numbers

plot_num_100(X_train[:100,:])
hidden_layer_sizes = [25]

model1 = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,

                                      verbose=False,

                                      tol=0.0000100,

                                      max_iter = 100)

model1 = model1.fit(X_train[:1000], y_train[:1000])
y_pred = model1.predict(X_test)

y_pred_train = model1.predict(X_train)

print(1 - np.sum(y_pred_train[:1000] != y_train[:1000]) / len(y_train[:1000]))

print(1 - np.sum(y_pred != y_test) / len(y_test))
for i in np.random.choice(len(X_test), 5):

    plot_num(X_test[i, :])

    _=plt.title('Prediction: {}'.format(y_pred[i]))