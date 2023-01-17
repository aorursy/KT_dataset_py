# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import torch
def grad(x, y):

    x.requires_grad_(True)

    y.requires_grad_(True)

    len_x = x.pow(2).sum().sqrt()

    len_y = y.pow(2).sum().sqrt()

    xy = x * y

    scalar_sum = xy.sum()

    res = scalar_sum/(len_x*len_y)

    res.backward()

    print("x", x.grad)

    print("y", y.grad)

x = torch.tensor([5.,8.,12.])

y = torch.tensor([10.,6.,2.])

grad(x,y)
import torch as tr

import numpy as np

import pandas as pd



df = pd.read_csv('../input/dataset_184_covertype.csv')

data = df.values
x_data = tr.tensor(data[:, :-1].astype(np.float32))

y_data = data[:, -1]



for i in range(10):

    x_data[:, i] -= x_data[:, i].mean()

    x_data[:, i] /= x_data[:, i].std()
from sklearn.preprocessing import OneHotEncoder



le = OneHotEncoder(sparse=False)

y_data = le.fit_transform(y_data.reshape(-1, 1))

y_data = tr.tensor(y_data).float()





y_data = [y.reshape(-1, 1) for y in y_data]

x_data = [x.reshape(-1, 1) for x in x_data]





from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=9)





from sklearn.utils import shuffle

from sklearn.metrics import classification_report
class Network:

    def __init__(self, data_size, use_gpu = False):

        

        self.epoch_count = 10

        self.batch_size = 200

        self.weights = [tr.randn(count, previous_count) for previous_count, count in zip([data_size[0], 100], [100, data_size[1]])]

        self.biases = [tr.zeros(count, 1) for count in [100, data_size[1]]]

        self.activation_funcs = [self.relu] + [self.logsoftmax]

        

    def relu(self, x):

        return tr.max(x, tr.tensor(0.))

    

    def logsoftmax(self, x):

        return tr.log_softmax(x, dim = 0, dtype = tr.float32)

    

    def batchify(self, unbatchified):

        x_batch = []

        y_batch = []

        for datapoint in unbatchified:

            x_data, y_data = datapoint

            x_batch.append(x_data.tolist())

            y_batch.append(y_data.tolist())

        

        return [m.squeeze().t() for m in [tr.tensor(x_batch), tr.tensor(y_batch)]]

        

    def feedforward(self, a):

        for w, b, af in zip(self.weights, self.biases, self.activation_funcs):

            a = af(tr.mm(w, a) + b)

        return a

    

    def loss(self, x, y_real):

        y = self.feedforward(x)

        batch_size = x.shape[1]

        return -((y_real * y).sum() / batch_size)

    

    def train(self, X_train, y_train, X_test, y_test, lr):

        

        for j in range(self.epoch_count):

            

            train_data = shuffle(list(zip(X_train, y_train)))

            

            for x_data, y_data in (self.batchify(train_data[k:k + self.batch_size]) for k in range(0, len(train_data), self.batch_size)):

                

                for w, b in zip(self.weights, self.biases):

                    w.requires_grad_(True)

                    b.requires_grad_(True)

        

                loss = self.loss(x_data, y_data)

                loss.backward()

        

                for w, b in zip(self.weights, self.biases):

                    w.requires_grad_(False)

                    b.requires_grad_(False)

                    w -= lr * w.grad

                    b -= lr * b.grad

                    w.grad.zero_()

                    b.grad.zero_()

            

            # count loss

            x, y_truth = self.batchify(list(zip(X_test, y_test)))

            y = self.feedforward(x)

            

            batch_size = x.shape[1]

            loss = -((y_truth * y).sum() / batch_size)

            

            print("Epoch {}/{} — loss {}".format((j+1), self.epoch_count, loss))

    

    def predict_test_data(self, network, X_test, y_test):

        test = tr.tensor([(tr.argmax(network.feedforward(x_test)), tr.argmax(y_test)) for (x_test, y_test) in list(zip(X_test, y_test))])

        return classification_report(test[:, 1], test[:, 0], target_names=le.get_feature_names())

        

    def predict(self, network, x_data):

        res = network.feedforward(x_data)

        return le.inverse_transform(res.reshape(1,-1))
network = Network([len(X_train[0]),len(y_train[0])])



network.train(X_train, y_train, X_test, y_test, 0.15)



report = network.predict_test_data(network, X_test, y_test)

print(report)
r_data = [np.random.rand(54).astype(np.float32) * 2 - 1]

r_data = np.array(r_data).reshape(-1,1)

np.around(r_data)

print(r_data)



pred_result = network.predict(network, tr.tensor(r_data))

        

print("predicted result = ", pred_result)