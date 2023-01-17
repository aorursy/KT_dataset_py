import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib
train = pd.read_csv('../input/Kannada-MNIST/train.csv')
train.head()
t_label = train['label']

t_data = train.drop('label',axis = 1)

a = t_data.iloc[1:2]

img = a.values.reshape(28,28)

plt.imshow(img, cmap = matplotlib.cm.binary,interpolation="nearest")
def one_hot(arrays):

    full = []

    a = 0

    for i in arrays:

        a = a + 1

        y_t = np.zeros(10,dtype = 'float')

        for j in range(0,10):

            if(abs(i - j) < 1e-4):

                y_t[j] = 1.0

                

        full.append(y_t)

    

    full = np.asarray(full)

    return full



y_t = one_hot(t_label)
def softmax(x):

    shiftx = x - np.max(x)

    exps = np.exp(shiftx)

    return exps / np.sum(exps)
class bp:

    def __init__(self,i_size,h_size,o_size,lr):

        self.i_size = i_size

        self.h_size = h_size

        self.o_size = o_size

        self.lr = lr

        

        self.w1 = 0.01 * np.random.randn(self.i_size,self.h_size)

        self.b1 = np.zeros((1,self.h_size))

        

        self.w2 = 0.01 * np.random.randn(self.h_size, self.o_size)

        self.b2 = np.zeros((1, self.o_size))

        

    def forword(self,inputs):

        self.inputs = inputs.reshape(784,1)

        self.h_layer = np.maximum(0, np.dot(self.inputs.T, self.w1) + self.b1)#relu

        self.score = np.dot(self.h_layer,self.w2) + self.b2

        self.probs = softmax(self.score)

    

    #cross entry

    def loss(self,label):

        label = label.reshape(10,1)

        i = range(0,self.probs.shape[0])

        L_i = -np.log(self.probs[i,label.astype(int)[i]])

        loss = 1 / L_i.shape[0] * np.sum(L_i)

        return loss

      

    def backword(self,label):

        dscores = self.probs - label.reshape(1,10)

        

        dw2 = np.dot(self.h_layer.T, dscores)

        db2 = np.sum(dscores, axis=0, keepdims=True)

        

        dhidden = np.dot(dscores, self.w2.T)

        dhidden[self.h_layer <= 0] = 0

        

        dw1 = np.dot(self.inputs, dhidden)

        db1 = np.sum(dhidden, axis=0, keepdims=True)

        

        self.w1 += -self.lr * dw1

        self.b1 += -self.lr * db1

        self.w2 += -self.lr * dw2

        self.b2 += -self.lr * db2
BP = bp(784,100,10,0.01)

epoch = 300
X = t_data.values
for i in range(0,epoch):

    shuffle_index = np.random.permutation(60000)

    x_train,y_train = X[shuffle_index],y_t[shuffle_index]

    for j in range(0,500):

        inputs = x_train[j] / 255

        BP.forword(inputs)

        BP.backword(y_train[j])
test = pd.read_csv('../input/Kannada-MNIST/test.csv')

x_test = test.drop('id',axis = 1).values
results = []

for i in range(0,len(x_test)):

    BP.forword(x_test[i])

    result = np.argmax(BP.probs)

    results.append(result)
ids = list(range(1,len(results) + 1))
sub = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')
sub['id'] = ids

sub['label'] = results
sub.to_csv('ktzs_nn.csv')
sub.head()