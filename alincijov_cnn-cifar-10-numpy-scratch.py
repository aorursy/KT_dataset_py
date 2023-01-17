import numpy as np

from keras.datasets import cifar10

from IPython.display import clear_output, display

import matplotlib.pyplot as plt

from time import sleep
(X_train, y_train) , (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(len(X_train), 3, 32, 32) / 255.0

X_test = X_test.reshape(len(X_test), 3, 32, 32) / 255.0
y_train = np.eye(10)[y_train]

y_test = np.eye(10)[y_test]
def initialise_param_lecun_normal(FILTER_SIZE, IMG_DEPTH, scale=1.0):

    fan_in = FILTER_SIZE * FILTER_SIZE * IMG_DEPTH

    stddev = scale * np.sqrt(1./fan_in)

    shape = (IMG_DEPTH, FILTER_SIZE, FILTER_SIZE)

    return np.random.normal(loc = 0,scale = stddev,size = shape) / 9
# Activation functions
def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)
def relu(x):

    x[x < 0] = 0

    return x
# Convolutional and Maxpool function
def conv(l, w, b, f, convd, filter, image):

    for jj in range(0, l):

        for i in range(0, w):

            for j in range(0, w):

                convd[jj,i,j] = np.sum(image[:,i:i+f,j:j+f] * filter[jj]) + b[jj]



    return convd
def maxpool(x, f, s):

    (l, w, w) = x.shape

    pool = np.zeros((l, (w-f)//s+1,(w-f)//s+1))

    for jj in range(0,l):

        for i in range(0, w, s):

            for j in range(0, w, s):

                pool[jj,i//2,j//2] = np.max(x[jj,i:i+f,j:j+f])

    return pool
def nanargmax(a):

	idx = np.argmax(a, axis=None)

	multi_idx = np.unravel_index(idx, a.shape)

	if np.isnan(a[multi_idx]):

		nan_count = np.sum(np.isnan(a))

		idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]

		multi_idx = np.unravel_index(idx, a.shape)

	return multi_idx
def forward(x, theta, convds, filters, f):

    l, l1, w1, w2, w3, b1, b2, b3 = theta

    f1, f2 = filters

    c1, c2 = convds



    m = np.array([conv(l1, w1, b1, f, c1, f1, x[i]) for i in range(len(x))])

    m = relu(m)



    n = np.array([conv(l2, w2, b2, f, c2, f2, m[i]) for i in range(len(m))])

    n = relu(n)



    # size 2, stride 2

    o = np.array([maxpool(n[i], 2, 2) for i in range(len(n))])



    # flatten

    flat = o.reshape((len(o), (w2//2) * (w2//2) *l2))



    r = flat.dot(w3) + b3



    probs = np.array([softmax(r[i]) for i in range(len(r))])



    return m, n, o, flat, r, probs
def dfilter_init(n, l2, l1, f):

    df, df_sub = [], []

    db, db_sub = [], []

    for _ in range(0, n):

        for x in range(0, l2):

            df_sub.append(np.zeros((l1,f,f)))

            db_sub.append(0)

        df.append(df_sub)

        db.append(db_sub)

        df_sub, db_sub = [], []

    

    return np.array(df), np.array(db)
def backward(x, y, theta, convds, filters, f):

    l, l1, w1, w2, w3, b1, b2, b3 = theta

    m, n, o, flat, r, probs = forward(x, theta, convds, filters, f)



    dout = probs - y.reshape(y.shape[0], 10)



    dw3 = flat.T.dot(dout)

    db3 = np.expand_dims(np.sum(dout, axis=0), axis=0)



    df = dout.dot(w3.T)



    dpool = df.T.reshape((x.shape[0], l2, w2//2, w2//2))

    dc2 = np.zeros((len(n), l2, w2, w2))



    for nn in range(len(n)):

        for jj in range(0,l):

            for i in range(0, w2, 2):

                for j in range(0, w2, 2):

                    (a,b) = nanargmax(n[nn][jj,i:i+2,j:j+2])

                    dc2[nn][jj, i+a, j+b] = dpool[nn][jj, i//2, j//2]



    dc2[n <= 0] = 0



    dc1 = np.zeros((len(m), l1, w1, w1))



    df2, db2 = dfilter_init(len(m), l2, l1, f)

    df1, db1 = dfilter_init(len(m), l1, l, f)



    for mm in range(len(m)):

        for jj in range(0, l2):

            for i in range(0, w2):

                for j in range(0, w2):

                    df2[mm][jj] += dc2[mm][jj, i, j] * m[mm][:, i:i+f, j:j+f]

                    dc1[mm][:, i:i+f, j:j+f] += dc2[mm][jj, i, j] * f2[jj]

            db2[mm][jj] = np.sum(dc2[mm][jj])



    dc1[m <= 0]=0



    for mm in range(len(m)):

        for jj in range(0, l1):

            for i in range(0, w1):

                for j in range(0, w1):

                    df1[mm][jj] += dc1[mm][jj, i, j] * x[mm][:, i:i+f, j:j+f]

            db1[mm][jj] = np.sum(dc1[mm][jj])



    return dc1, dc2, df1, df2, dw3, db1, db2, db3
def average_grads(grads):

    return [np.average(grads[i], axis=0) for i in range(len(grads))]
def optimize(grads, theta, convds, filters, lr=0.01):

    dc1, dc2, df1, df2, dw3, db1, db2, db3 = grads

    l, l1, w1, w2, w3, b1, b2, b3 = theta

    c1, c2 = convds

    f1, f2 = filters



    c1 -= dc1 * lr

    c2 -= dc2 * lr



    f1 -= df1 * lr

    f2 -= df2 * lr



    w3 -= dw3 * lr



    b1 -= db1 * lr

    b2 -= db2 * lr

    b3 -= db3 * lr



    grads = dc1, dc2, df1, df2, dw3, db1, db2, db3

    theta = l, l1, w1, w2, w3, b1, b2, b3

    convds = c1, c2

    filters = f1, f2



    return grads, theta, convds, filters
def cross_entropy(predictions, targets, epsilon=1e-12):

    """

    Computes cross entropy between targets (encoded as one-hot vectors)

    and predictions. 

    Input: predictions (N, k) ndarray

           targets (N, k) ndarray        

    Returns: scalar

    """

    predictions = np.clip(predictions, epsilon, 1. - epsilon)

    N = predictions.shape[0]

    ce = -np.sum(targets*np.log(predictions+1e-9))/N

    return ce
# because CPU convolutional takes forever to train

# we select : 5 idx where the number is 9 and 5 idx for number 1

ten_idx = np.where(np.argmax(y_train.reshape(50000, 10), axis=1) == 9)[0][:5]

one_idx = np.where(np.argmax(y_train.reshape(50000, 10), axis=1) == 1)[0][:5]
features = np.concatenate((X_train[ten_idx], X_train[one_idx]), axis=0)

labels = np.concatenate((y_train[ten_idx], y_train[one_idx]), axis=0)
features.shape, labels.shape
np.random.seed(2342342)
NUM_FILT1 = 16

NUM_FILT2 = 16



IMG_DEPTH = 3

FILTER_SIZE = 5
## Initializing all the parameters

f1, f2, b1, b2 = [], [], [], []
for i in range(0, NUM_FILT1):

	f1.append(initialise_param_lecun_normal(FILTER_SIZE,IMG_DEPTH))

	b1.append(0.)



for i in range(0, NUM_FILT2):

	f2.append(initialise_param_lecun_normal(FILTER_SIZE,NUM_FILT1))

	b2.append(0.)



f1, f2, b1, b2 = np.array(f1), np.array(f2), np.array(b1), np.array(b2)
(l, w, w) = X_train[0].shape		

l1, l2 = len(f1), len(f2)

( _, f, f) = f1[0].shape

w1 = w-f+1

w2 = w1-f+1
c1 = np.zeros((l1, w1, w1))

c2 = np.zeros((l2, w2, w2))



w3 = np.random.normal(size=(2304, 10)) / 9

b3 = np.random.normal(size=(1, 10)) / 9



theta = l, l1, w1, w2, w3, b1, b2, b3

convds = c1, c2

filters = f1, f2
losses = []

for epoch in range(501):

    grads = backward(features, labels, theta, convds, filters, 5)



    grads = average_grads(grads)

    grads, theta, convds, filters = optimize(grads, theta, convds, filters, lr=0.1)



    if(epoch % 25 == 0):

        out = forward(features, theta, convds, filters, 5)[-1]

        loss = cross_entropy(out, labels)

        losses.append(loss)

        print(f'Epoch:%4d, Loss:%.3f' % (epoch, loss))
plt.plot(losses)
for i in range(0, len(features)):

    m, n, o, flat, r, probs = forward(features[i].reshape(1, 3, 32, 32), theta, convds, filters, 5)



    print('idx:{0}, Probs:{1}, y:{2}, cross:{3}'.format(i, np.argmax(probs), np.argmax(labels[i]), cross_entropy(probs, labels[i])))