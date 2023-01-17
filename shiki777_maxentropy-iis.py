import time

import numpy as np

from collections import defaultdict,Counter
def loadData(filename):

    fr = open(filename,'r')

    x,y = [],[]

    for line in fr.readlines():

        curline = line.strip().split(',')

        if int(curline[0]) == 0 or int(curline[0]) == 1:

            x.append([int(int(num) >= 128) for num in curline[1:]])

            y.append(int(curline[0]))

    x = np.array(x)

    y = np.array(y)

    return x,y
class MaxEntropy_IIS:

    def __init__(self,x_train,y_train,x_val,y_val):

        self.x_train = x_train

        self.y_train = y_train

        self.x_val = x_val

        self.y_val = y_val

        self.label_num = 2

        self.id2key,self.key2id,self.empir_dist = self.get_Empirical_Distribution()

        self.empir_dist = np.array(list(self.empir_dist.values()))

        self.w = np.zeros(self.empir_dist.shape[0])



    def get_Empirical_Distribution(self,):

        E = defaultdict(int)

        id2key,key2id = {},{}

        for i in range(self.x_train.shape[0]):

            for j in range(self.x_train.shape[1]):

                if (j,self.x_train[i][j],self.y_train[i]) not in key2id:

                    id2key[len(id2key)] = (j,self.x_train[i][j],self.y_train[i])

                    key2id[(j,self.x_train[i][j],self.y_train[i])] = len(key2id)

                E[key2id[(j,self.x_train[i][j],self.y_train[i])]] += 1 / self.x_train.shape[0]

        return id2key,key2id,E

        

    def get_Model_Distribution(self,):

        E = np.zeros(self.empir_dist.shape[0])

        for i in range(self.x_train.shape[0]):

            P = [self.get_Conditional_Probability(self.x_train[i],label) for label in range(self.label_num)]

            for j in range(self.x_train.shape[1]):

                for label in range(self.label_num):

                    if (j,self.x_train[i][j],label) in self.key2id:

                        E[self.key2id[(j,self.x_train[i][j],label)]] += 1/self.x_train.shape[0] * P[label]

        return E

        

    def get_Conditional_Probability(self,x,y):

        z,Z = 0,0

        for i in range(self.x_train.shape[1]):

            if (i,x[i],y) in self.key2id:

                z += self.w[self.key2id[(i,x[i],y)]]

        for label in range(self.label_num):

            sum_ = 0

            for i in range(self.x_train.shape[1]):

                if (i,x[i],label) in self.key2id:

                    sum_ += self.w[self.key2id[(i,x[i],label)]]

            Z += np.exp(sum_)

        z = np.exp(z)

        return z / Z

    

    def fit(self,epochs):

        M = np.sum(self.empir_dist) * self.x_train.shape[0]

        for epoch in range(epochs):

            start = time.time()

            E = self.get_Model_Distribution()

            sigma = [0] * len(self.w)

            for i in range(len(sigma)):

                sigma[i] = 1 / M * np.log(self.empir_dist[i] / E[i])

            self.w = [self.w[idx] + sigma[idx] for idx in range(len(self.w))]

            if True:

                acc = self.test(self.x_train,self.y_train)

                print("Epoch {} Time costs {:.2f} The accuracy {:.3f}".format(epoch + 1,time.time() - start,acc))

            else:

                print("Epoch {} Time costs {:.2f}".format(epoch + 1,time.time() - start))

                

    def test(self,x_val,y_val):

        correct = 0

        for i in range(x_val.shape[0]):

            P = [self.get_Conditional_Probability(x_val[i],label) for label in range(self.label_num)]

            idx = P.index(max(P))

            if idx == y_val[i]: correct += 1

        return correct / x_val.shape[0]
x_train_mnist,y_train_mnist = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')

x_val_mnist,y_val_mnist = loadData('/kaggle/input/mnist-percetron/mnist_test.csv')
Counter(y_train_mnist)
Counter(y_val_mnist)
model = MaxEntropy_IIS(x_train_mnist,y_train_mnist,x_val_mnist,y_val_mnist)
model.test(x_train_mnist,y_train_mnist)
model.test(x_val_mnist,y_val_mnist)
model.fit(1)
model.test(x_train_mnist,y_train_mnist)
model.test(x_val_mnist,y_val_mnist)