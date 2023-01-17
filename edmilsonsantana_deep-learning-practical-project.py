import tensorflow as tf

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import random

from sklearn.utils import shuffle
def getData(balance_ones=True):

    Y = []

    X = []

    first = True

    for line in open('../input/fer20131.csv'):

        if first:

            first = False

        else:

            row = line.split(',')

            Y.append(int(row[0]))

            X.append([int(p) for p in row[1].split()])



    X, Y = np.array(X) / 255.0, np.array(Y)

    

    if balance_ones:

        X0, Y0 = X[Y!=1, :], Y[Y!=1]

        X1 = X[Y==1, :]

        X1 = np.repeat(X1, 9, axis=0)

        X = np.vstack([X0, X1])

        Y = np.concatenate((Y0, [1]*len(X1)))



    return X, Y



def getImageData():

    X, Y = getData()

    N, D = X.shape

    d = int(np.sqrt(D))

    X = X.reshape(N, d, d, 1)

    return X, Y
X, Y = getImageData()
labels = list(set(Y))



fig = plt.figure(figsize=(15, 20))

columns = 5

rows = 5



for i in labels:

    imagens = X[Y==i]

    qtd_imagens = len(imagens)

    imagem = imagens[random.randint(0, qtd_imagens), :].reshape((48, 48))

    ax = fig.add_subplot(rows, columns, i + 1)

    ax.set_title('%d' % (i))

    plt.imshow(imagem, cmap='gray')

plt.show()
def init_weight_and_bias(M1, M2):

    W = np.random.randn(M1, M2) / np.sqrt(M1)

    b = np.zeros(M2)

    return W, b
class HiddenLayer(object):

    def __init__(self, M1, M2, activation=None):

        self.M1 = M1

        self.M2 = M2

        self.activation = activation

        W, b = init_weight_and_bias(M1, M2)

        self.W = tf.Variable(W.astype(np.float32))

        self.b = tf.Variable(b.astype(np.float32))

        self.params = [self.W, self.b]

    

    def forward(self, X, is_training):

        act_value = tf.matmul(X,self.W) + self.b

        if self.activation is not None:

            act_value = self.activation(act_value)

        return act_value
class ANN(object):

    def __init__(self, hidden_layer_sizes, layer_class=HiddenLayer):

        self.hidden_layer_sizes = hidden_layer_sizes

        self.layer_class = layer_class

    

    def build_Layers(self, X, Y, activation):

        _, D = X.shape

        K = len(set(Y))

        

        self.layers = []

        M1 = D

        for M2 in self.hidden_layer_sizes:

            h = self.layer_class(M1, M2, activation)

            self.layers.append(h)

            M1 = M2

        

        h = HiddenLayer(M1, K)

        self.layers.append(h)

        

        self.params = []

        for h in self.layers:

            self.params += h.params

        

        return (None, D)

    

    def train(self, inputs, labels, Xtrain, Ytrain, Xvalid, Yvalid, epochs, n_batches, batch_sz, print_period):

        costs = []

        init = tf.global_variables_initializer()

        with tf.Session() as session:

            session.run(init)

            for i in range(epochs):

                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)

                for j in range(n_batches):

                    Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz)]

                    Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz)]

                    session.run(self.train_op, feed_dict={inputs: Xbatch, labels: Ybatch})

                    

                    if ((j + 1) % print_period == 0):

                        c = session.run(self.cost_op, feed_dict={inputs: Xvalid, labels: Yvalid})

                        p = session.run(self.predict_op, feed_dict={inputs: Xvalid})

                        costs.append(c)

                        acc = np.mean(p != Yvalid)

                        print("i:", i, "j:", j, "nb:", n_batches, "cost:", c, "error rate:", acc)

        return costs

    

    def fit(self, X, Y, activation=tf.nn.relu, learning_rate=1e-3, reg=1e-3,mu=0.99, decay=0.99999, print_period=20, epochs=20, batch_sz=100, show_fig=False):

        learning_rate = np.float32(learning_rate)

        mu = np.float32(mu)

        reg = np.float32(reg)

        decay = np.float32(decay)

        

        X = X.astype(np.float32)

        Y = Y.astype(np.int32)

        X, Y = shuffle(X, Y)

        Xvalid  = X[-1000:]

        Yvalid  = Y[-1000:]

        Xtrain = X[:-1000]

        Ytrain = Y[:-1000]

        

        N = Xtrain.shape[0]

        

        input_shape = self.build_Layers(X, Y, activation)

        

        if batch_sz is None:

            batch_sz = N

            

        inputs = tf.placeholder(tf.float32, shape=input_shape, name='inputs')

        labels = tf.placeholder(tf.int32, shape=(None,), name='labels')

        logits = self.forward(inputs, is_training=True)

        

        self.cost_op = tf.reduce_mean(

            tf.nn.sparse_softmax_cross_entropy_with_logits(

                logits=logits,

                labels=labels

            )

        )

        

        if(reg is not None):

            rcost = reg*sum([tf.nn.l2_loss(p) for p in self.params])

            self.cost_op += rcost

        

        self.train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(self.cost_op)

        #self.train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.9, use_nesterov=True).minimize(self.cost_op)



        self.predict_op = self.predict(inputs)

        

        n_batches = N // batch_sz

        

        costs = self.train(inputs, labels, Xtrain, Ytrain, Xvalid, Yvalid, 

                           epochs, n_batches, batch_sz, print_period)

       

        if show_fig:

            plt.plot(costs)

            plt.show()

            

    def forward(self, X, is_training):

        out = X

        for h in self.layers:

            out = h.forward(out, is_training)

        return out

    

    def predict(self, X):

        pY = self.forward(X, is_training=False)

        return tf.argmax(pY, 1)

        

        
class ConvPoolLayer(object):

    def __init__(self, filter_width, filter_height, feature_in, feature_out, pool_sz=(2,2)):

        self.pool_sz = pool_sz

        self.shape = (filter_width, filter_height, feature_in, feature_out)

        self.init_filter()

        

    def init_filter(self):

        W_init = np.random.randn(*self.shape) * np.sqrt(2) / np.sqrt(np.prod(self.shape[:-1]) + self.shape[-1]*np.prod(self.shape[:-2] / np.prod(self.pool_sz)))

        b_init = np.zeros(self.shape[-1], dtype=np.float32)

        self.W = tf.Variable(W_init.astype(np.float32))

        self.b = tf.Variable(b_init)

        self.params = [self.W, self.b]

        

    def convpool(self, X):

        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')

        conv_out = tf.nn.bias_add(conv_out, self.b)

        ksize = [1, self.pool_sz[0], self.pool_sz[1], 1]

        pool_out = tf.nn.max_pool(conv_out, ksize=ksize, strides=ksize, padding='SAME')

        return pool_out

        

        
class CNN(ANN):

    def __init__(self, hidden_layer_sizes, convpool_layer_sizes):

        ANN.__init__(self, hidden_layer_sizes)

        self.convpool_layer_sizes = convpool_layer_sizes

        



    def build_Layers(self, X, Y, activation):

        _, H, W, C = X.shape

        pool_sz = (2, 2)

        K = len(set(Y))

        

        self.convpool_layers = []

        self.params = []

        self.layers = []

        

        feature_in = C

        for feature_out, filter_w, filter_h in self.convpool_layer_sizes:

            layer = ConvPoolLayer(filter_w, filter_h, feature_in, feature_out, pool_sz)

            self.params += layer.params

            self.convpool_layers.append(layer)

            feature_in = feature_out

            

        M1 = feature_in * ((H // (len(self.convpool_layers) * pool_sz[0])) * 

                           (W // (len(self.convpool_layers) * pool_sz[1])))

       

        for M2 in self.hidden_layer_sizes:

            layer = self.layer_class(M1, M2, activation)

            self.params += layer.params

            self.layers.append(layer)

            M1 = M2

        

        h = HiddenLayer(M1, K)

        self.params += h.params

        self.layers.append(h)

        

        

        return (None, H, W, C)

    

    def forward(self, X, is_training):

        out = X

        for layer in self.convpool_layers:

            out = layer.convpool(out)

        

        out_shape = out.get_shape().as_list()

        out = tf.reshape(tensor=out, shape=[-1, np.prod(out_shape[1:])])

        

        for layer in self.layers:

            out = layer.forward(out, is_training)

            

        return out
## Experiment 1



X, Y = getData()



model = ANN([2000, 1000, 500])

model.fit(X, Y, show_fig=True, batch_sz=100, epochs=10, learning_rate=1e-2, decay=0.999)
## Experiment 2



X, Y = getImageData()



model = CNN(convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],

           hidden_layer_sizes=[500, 300])

model.fit(X, Y, show_fig=True, batch_sz=30, epochs=3)