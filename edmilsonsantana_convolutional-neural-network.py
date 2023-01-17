import pandas as pd

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from scipy.signal import convolve2d

import wave

import sys

from scipy.io.wavfile import write
spf = wave.open("../input/helloworld.wav")
signal = spf.readframes(-1);

signal = np.fromstring(signal, 'Int16')
print("Numpy Signal Shape: ", signal.shape);
plt.plot(signal)

plt.title('Hello World without echo');
delta = np.array([1., 0, 0])

noecho = np.convolve(signal, delta)

print("Noecho signal:", noecho.shape)

assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.0000001)
noecho = noecho.astype(np.int16)

write('noecho.wav', 16000, noecho)
filt = np.zeros(16000)

filt[0] = 1

filt[4000]  = 0.6

filt[8000]  = 0.3

filt[12000] = 0.2

filt[15999] = 0.1
out = np.convolve(signal, filt)

out = out.astype(np.int16)

write('out.wav', 16000, out)
img = mpimg.imread('../input/lena.png')
plt.imshow(img)
bw = img.mean(axis=2)
plt.imshow(bw, cmap='gray')
W = np.zeros((20, 20))



for i in range(20):

    for j in range(20):

        dist = (i - 9.5) ** 2 + (j - 9.5) ** 2

        W[i, j] = np.exp(-dist / 50)
plt.imshow(W, cmap='gray')
out = convolve2d(bw, W)

plt.imshow(out, cmap='gray')
out.shape
out = convolve2d(bw, W, mode='same')

plt.imshow(out, cmap='gray')
out.shape
out3 = np.zeros(img.shape)

for i in range(3):

    out3[:, :, i] = convolve2d(img[:, :, i], W, mode='same')

out3 /= out3.max()
plt.imshow(out3)
Hx = np.array([[-1.,  0., 1.],

               [-2.,  0., 2.],

               [-1.,  0., 1.]],dtype=np.float32)

Hy = Hx.T
Gx = convolve2d(bw, Hx)
plt.imshow(Gx, cmap='gray')
Gy = convolve2d(bw, Hy)
plt.imshow(Gy, cmap='gray')
G = np.sqrt(Gx*Gx + Gy*Gy)
plt.imshow(G, cmap='gray')
def convolve(X, W):

    x_i, x_j = X.shape

    w_i, w_j = W.shape

    Y = np.zeros((x_i + w_i - 1, x_j + w_j - 1))

    

    for i in range(x_i):

        for j in range(x_j):

            Y[i:i+w_i, j:j+w_j] += X[i, j] * W

        

    return Y

    
plt.imshow(convolve(bw, W), cmap='gray')
import theano

import theano.tensor as T

from theano.tensor.nnet import conv2d

from theano.tensor.nnet import relu

from theano.tensor.signal.pool import pool_2d

from scipy.io import loadmat

from sklearn.utils import shuffle

from datetime import datetime

import tensorflow as tf
def error_rate(p, t):

    return np.mean(p != t)
def y2indicator(y):

    N = len(y)

    ind = np.zeros((N, 10))

    for i in range(N):

        ind[i, y[i]] = 1

    return ind
train = loadmat('../input/train_32x32.mat')

test = loadmat('../input/test_32x32.mat')
def convpool(X, W, b):

    conv_out = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')

    conv_out = tf.nn.bias_add(conv_out, b)

    pool_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2 ,1], padding='SAME')

    return pool_out
def init_filter(shape, poolsz):

    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[:-1]) + shape[-1]*np.prod(shape[:-2] / np.prod(poolsz)))

    return w.astype(np.float32)
def rearrange(X):

    N = X.shape[-1]

    out = np.zeros((N, 32, 32, 3), dtype=np.float32)

    for i in range(N):

        for j in range(3):

            out[i, :, :, j] = X[:, :, j, i]

    return out / 255
Xtrain = rearrange(train['X'])

Ytrain = train['y'].flatten() - 1

Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

Ytrain_ind = y2indicator(Ytrain)
Xtest = rearrange(test['X'])

Ytest = test['y'].flatten() - 1

Ytest_ind = y2indicator(Ytest)
max_iter = 6

print_period = 10

lr = np.float32(0.0001)

reg = np.float32(0.01)

mu = np.float32(0.99)

N = Xtrain.shape[0]

batch_sz = 500

n_batches = int(N / batch_sz)
Xtrain = Xtrain[:73000, :]

Ytrain = Ytrain[:73000]

Xtest = Xtest[:26000, :]

Ytest = Ytest[:26000]
M = 500

K = 10

poolsz = (2,2)



# Filtro 1

W1_shape = (5, 5, 3, 20)

W1_init = init_filter(W1_shape, poolsz)

b1_init = np.zeros(W1_shape[-1], dtype=np.float32)

# Filtro 2

W2_shape = (5, 5, 20, 50)

W2_init = init_filter(W2_shape, poolsz)

b2_init = np.zeros(W2_shape[-1], dtype=np.float32)



# Primeira camada rede vanilla

W3_init = np.random.randn(W2_shape[-1] * 8 * 8, M) / np.sqrt(W2_shape[-1] * 8 * 8 + M)

b3_init = np.zeros(M, dtype=np.float32)



# Segunda camada rede vanilla

W4_init = np.random.randn(M, K) / np.sqrt(M + K)

b4_init = np.zeros(K, dtype=np.float32)
X = tf.placeholder(tf.float32, shape=(batch_sz, 32, 32, 3), name='X')

T = tf.placeholder(tf.float32, shape=(batch_sz, K), name='T')

W1 = tf.Variable(W1_init.astype(np.float32))

b1 = tf.Variable(b1_init.astype(np.float32))

W2 = tf.Variable(W2_init.astype(np.float32))

b2 = tf.Variable(b2_init.astype(np.float32))

W3 = tf.Variable(W3_init.astype(np.float32))

b3 = tf.Variable(b3_init.astype(np.float32))

W4 = tf.Variable(W4_init.astype(np.float32))

b4 = tf.Variable(b4_init.astype(np.float32))
Z1 = convpool(X, W1, b1)

Z2 = convpool(Z1, W2, b2)

Z2_shape = Z2.get_shape().as_list()

Z2r = tf.reshape(Z2, [Z2_shape[0], np.prod(Z2_shape[1:])])

Z3 = tf.nn.relu(tf.matmul(Z2r, W3) + b3)

Yish = tf.matmul(Z3, W4) + b4
cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))

train_op = tf.train.RMSPropOptimizer(0.0001, decay=0.99, momentum=0.9).minimize(cost)

predict_op = tf.argmax(Yish, 1)
start_time = datetime.now()

costs = []

init = tf.initialize_all_variables()

filters = []

with tf.Session() as session:

    session.run(init)

    

    for i in range(max_iter):

        for j in range(n_batches):

            Xbatch = Xtrain[j*batch_sz: (j*batch_sz + batch_sz)]

            Ybatch = Ytrain_ind[j*batch_sz: (j*batch_sz + batch_sz)]



            if(len(Xbatch) == batch_sz):

                session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

                if(j% print_period == 0):

                    test_cost = 0

                    prediction = np.zeros(len(Xtest))

                    for k in range(int(len(Xtest) / batch_sz)):

                        Xtest_batch = Xtest[k*batch_sz: (k*batch_sz + batch_sz)]

                        Ytest_batch = Ytest_ind[k*batch_sz: (k*batch_sz + batch_sz)]

                        test_cost += session.run(cost, feed_dict={X: Xtest_batch, T: Ytest_batch})

                        prediction[k*batch_sz: (k*batch_sz + batch_sz)] = session.run(

                            predict_op, feed_dict={X: Xtest_batch})

                        err = error_rate(prediction, Ytest)

                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))

                    costs.append(test_cost)

    filters = [session.run(W1), session.run(W2)]

    

print("Elapsed Time:", (datetime.now() - start_time))

plt.plot(costs)
def plot_filter(kernel, plot_shape, title_text='kernel'):

    h, w, color_channel, feature_map = kernel.shape

    grid = np.zeros((plot_shape[0] * h, plot_shape[1] * w))

    m = 0

    n = 0

    for i in range(feature_map):

        for j in range(color_channel):

            img = kernel[:, :, j, i]

            grid[m*h:(m + 1) * h, n*w:(n + 1) * w] = img

            m += 1

            if m >= plot_shape[0]:

                m = 0

                n += 1

    plt.imshow(grid, cmap='gray', interpolation='nearest', aspect='auto')

    plt.title(title_text)

    plt.show()
plot_filter(filters[0], (8,8), 'W1')
plot_filter(filters[1], (32,32), 'W2')