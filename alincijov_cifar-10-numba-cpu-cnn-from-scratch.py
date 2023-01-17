import numpy as np

import numba as na

from numba import cuda 



from tqdm import tqdm



from keras.datasets import cifar10

from IPython.display import clear_output, display

import matplotlib.pyplot as plt

from time import sleep
(X_train, y_train) , (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(len(X_train), 3, 32, 32) / 255.0

X_test = X_test.reshape(len(X_test), 3, 32, 32) / 255.0
y_train = np.eye(10)[y_train]

y_test = np.eye(10)[y_test]
@na.jit(nopython=True)

def maxpool(x, f, s):

    (l, w, w) = x.shape

    pool = np.zeros((l, (w-f)//s+1,(w-f)//s+1))

    for jj in range(0,l):

        for i in range(0, w, s):

            for j in range(0, w, s):

                pool[jj,i//2,j//2] = np.max(x[jj,i:i+f,j:j+f])

    return pool
@na.jit(nopython=True)

def relu(x):

    return x * (x > 0)
@na.jit(nopython=True)

def softmax(x):

    return np.exp(x) / np.sum(np.exp(x), axis=0)
@na.jit(nopython=True)

def Huber(yHat, y, delta=1.):

    return np.where(np.abs(y-yHat) < delta,.5*(y-yHat)**2 , delta*(np.abs(y-yHat)-0.5*delta))
@na.jit(nopython=True)

def forward(image, theta):

    f1, f2, b1, b2, b3, t3 = theta



    (l, w, w) = image.shape

    l1, l2 = len(f1), len(f2)

    (_, f, f) = f1[0].shape



    w1 = w - f + 1

    w2 = w1 - f + 1

    sizes = l, f, l1, l2,  w1, w2



    conv1 = np.zeros((l1, w1, w1))

    conv2 = np.zeros((l2, w2, w2))



    for jj in range(0, l1):

        for x in range(0, w1):

            for y in range(0, w1):

                conv1[jj, x, y] = np.sum(image[:,x:x+f,y:y+f] * f1[jj]) + b1[jj]

    

    conv1 = relu(conv1) 



    for jj in range(0, l2):

        for x in range(0, w2):

            for y in range(0, w2):

                conv2[jj,x,y] = np.sum(conv1[:,x:x+f,y:y+f] * f2[jj]) + b2[jj]



    conv2 = relu(conv2)



    pooled_layer = maxpool(conv2, 2, 2)	

    fc1 = pooled_layer.reshape(((w2//2)*(w2//2)*l2,1))



    out = softmax(t3.dot(fc1) + b3).T



    gamma = conv1, conv2, fc1



    return gamma, sizes, out
@na.jit(nopython=True)

def nanargmax(z):



    a, b = 0, 0

    max_nr = 0.0



    for i in range(len(z)):

        for j in range(len(z[0])):

            if not np.isnan(z[i, j]):

                if(z[i, j] > max_nr):

                    max_nr = z[i, j]

                    a, b = i, j

    return (a, b)
@na.jit(nopython=True)

def backward(image, label, gamma, sizes, theta):

    f1, f2, b1, b2, b3, t3 = theta

    l, f, l1, l2,  w1, w2 = sizes

    conv1, conv2, fc1 = gamma



    dout = out - label



    dt3 = fc1.dot(dout).T

    db3 = np.sum(dout.T, axis=1).reshape(10, 1)



    dfc1 = dout.dot(t3).T

    dpool = dfc1.T.reshape((l2, w2//2, w2//2))



    dconv2 = np.zeros((l2, w2, w2))



    for jj in range(0,l2):

        for i in range(0, w2):

            for j in range(0, w2):

                (a,b) = nanargmax(conv2[jj,i:i+2,j:j+2])

                dconv2[jj,i+a,j+b] = dpool[jj,i//2,j//2]



    dconv2 = relu(dconv2)



    dconv1 = np.zeros((l1, w1, w1))



    df1 = np.zeros(((l1, l, f, f)))

    db1 = np.zeros((l1))



    df2 = np.zeros((l1, l2, f, f))

    db2 = np.zeros((l2))



    for jj in range(0, l2):

        for x in range(0, w2):

            for y in range(0, w2):

                df2[jj] += dconv2[jj,x,y] * conv1[:,x:x+f,y:y+f]

                dconv1[:,x:x+f,y:y+f] += dconv2[jj,x,y] * f2[jj]

        db2[jj] = np.sum(dconv2[jj])



    for jj in range(0, l1):

        for x in range(0, w1):

            for y in range(0, w1):

                df1[jj] += dconv1[jj,x,y] * image[:,x:x+f,y:y+f]



        db1[jj] = np.sum(dconv1[jj])



    return df1, df2, db1, db2, db3, dt3
def init_theta():



    np.random.seed(234234232)



    stddev = np.sqrt(1. / (5 * 5 * 3))



    f1 = np.random.normal(loc=0, scale=stddev, size=(16, 3, 5, 5))

    f2 = np.random.normal(loc=0, scale=stddev, size=(16, 16, 5, 5))

    b1 = np.random.normal(loc=0, scale=stddev, size=(16))

    b2 = np.random.normal(loc=0, scale=stddev, size=(16))

    b3 = np.random.normal(loc=0, scale=stddev, size=(10, 1))

    t3 = np.random.normal(loc=0, scale=stddev, size=(10, 2304))



    return f1, f2, b1, b2, b3, t3
def init_momentum(theta):

    f1, f2, b1, b2, b3, t3 = theta



    bv1 = np.zeros((b1.shape))

    v1 = np.zeros((f1.shape))



    bv2 = np.zeros((b2.shape))

    v2 = np.zeros((f2.shape))



    bv3 = np.zeros((b3.shape))

    v3 = np.zeros((t3.shape))



    return bv1, bv2, bv3, v1, v2, v3
def init_grads(theta):

    f1, f2, b1, b2, b3, t3 = theta



    db1 = np.zeros((b1.shape))

    df1 = np.zeros((f1.shape))



    db2 = np.zeros((b2.shape))

    df2 = np.zeros((f2.shape))



    db3 = np.zeros((b3.shape))

    dt3 = np.zeros((t3.shape))



    return df1, df2, db1, db2, db3, dt3
# momentum

MO = 0.05



# parameters

lr = 0.1

decay = 0.001

batch_size = 100
theta = init_theta()



losses = []
for epoch in range(100):



    # [0] df1, [1] df2, [2] db1, [3] db2, [4] db3, [5] dt3

    temp_grads = init_grads(theta)



    # init momentum properties

    bv1, bv2, bv3, v1, v2, v3 = init_momentum(theta)



    # clear output

    clear_output(wait=True)



    # use only the first 1000 elements in X_train

    for batch, idx in enumerate(np.array_split(np.arange(1000), (len(np.arange(1000))/100))):



        all_grads = []

        acc = 0



        for i in idx:

            gamma, sizes, out = forward(X_train[i], theta)

            if np.argmax(out)==np.argmax(y_train[i]):

                acc += 1

            grads = backward(X_train[i], y_train[i], gamma, sizes, theta)

            all_grads.append(grads)



        f1, f2, b1, b2, b3, t3 = theta



        all_grads = np.array(all_grads)[0]

        temp_grads = tuple([temp_grads[i] + all_grads[i] for i in range(len(all_grads))])



        # learning decay

        # lr *= (1. / (1. + decay * (epoch + 1)))



        for i in range(len(df1)):

            v1[i] = MO * v1[i] - lr * temp_grads[0][i] / batch_size

            f1[i] += v1[i]



            v2[i] = MO * v2[i] - lr * temp_grads[1][i] / batch_size

            f2[i] += v2[i]



            bv1[i] = MO * bv1[i] -  lr * temp_grads[2][i] / batch_size

            b1[i] += bv1[i]



            bv2[i] = MO * bv2[i] -  lr * temp_grads[3][i] / batch_size

            b2[i] += bv2[i]



        bv3 = MO * bv3 - lr * temp_grads[4] / batch_size

        b3 += bv3



        v3 = MO * v3 - lr * temp_grads[5] / batch_size

        t3 += v3



        theta = f1, f2, b1, b2, b3, t3





        nr = np.random.randint(low=0, high=1000, size=1)[0]

        gamma, sizes, out = forward(X_train[nr], theta)

        loss = np.mean(Huber(out,y_train[nr]))

        losses.append(loss)

        print('  ---- Epoch:{0:3d}  ---- Batch:{1:2.0f}/{2:1.0f}  ---- Learning_rate:{3:1.4f}  ---- Acc:{4:3d}/{5:3d}  ---- Loss:{6:3.5f}  ---- '

                .format((epoch + 1), (batch + 1), (len(np.arange(1000)) / 100), lr, acc, len(idx), loss))