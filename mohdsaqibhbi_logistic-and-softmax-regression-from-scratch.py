import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import pandas as pd



import warnings

warnings.filterwarnings('ignore')
class LogisticsRegression:

    

    def sigmoid(self, x):

    

        # shape(samples, 1)

        z = ((np.dot(x, self.weight)) + self.bias)



        # shape(samples, 1)

        return (1 / (1 + np.exp(-z)))

        

    def forward(self, x):

        

        # shape(samples, 1)

        return self.sigmoid(x)

    

    def binary_crossEntropy(self, y, y_hat):

        

        # shape(samples, 1)

        return ((-1) * y * (np.log(y_hat))) - ((1 - y) * (np.log(1 - y_hat)))

        

    def cost(self, y, y_hat):

        

        # scalar

        return np.mean(self.binary_crossEntropy(y, y_hat))

        

    def train(self, x, y, alpha, epoch, random_state=-1):

        

        # x : shape(#samples, #features)

        # y : shape(#samples, 1)

        

        m, n = x.shape[0], x.shape[1]

        

        if random_state != -1:

            np.random.seed(random_state)

        

        # shape(#features, 1)

        self.weight = np.random.randn(n,1)



        # shape(1,1)

        self.bias = np.zeros((1,1))

        

        self.epoch = epoch

        

        self.cost_list = []

        

        for i in range(self.epoch):

            

            # shape(#samples, 1)

            y_hat = self.forward(x)

    

            # scalar

            loss = self.cost(y, y_hat)



            self.cost_list.append(loss)



            # Gradient

            # dL_dw : dLoss/dweight (#features, 1)

            dL_dw = (np.dot(x.T, (y_hat - y)))/m



            # dL_db : dLoss/dbias (1, 1)

            dL_db = np.sum((y_hat - y)/m)



            # shape(#features, 1)

            self.weight = self.weight - (alpha * dL_dw)



            # shape(1, 1)

            self.bias = self.bias - (alpha * dL_db)

            

    def plot_convergence(self):

        

        plt.plot([i for i in range(self.epoch)], self.cost_list)

        plt.xlabel('Epochs'); plt.ylabel('Binary Cross Entropy')

        

    def predict(self, x_test):

        

        # shape(samples, 1)

        y_hat = self.forward(x_test)

        return np.where(y_hat>=0.5, 1, 0)
def randomDataset(m, n, random_state=-1):

    

    if random_state != -1:

        np.random.seed(random_state)

        

    x = np.random.randn(m, n)

    slope = np.random.randn(n, 1)

    epsilon = np.random.randn(1, 1)

    y = (1 / (1 + np.exp(-(np.dot(x, slope) + epsilon))))

    y = np.where(y>=0.5, 1, 0)

    print(slope, epsilon)

    

    return x, y
def train_test_split(x, y, size=0.2, random_state=-1):

    

    if random_state != -1:

        np.random.seed(random_state)

        

    x_val = x[:int(len(x)*size)]

    y_val = y[:int(len(x)*size)]

    x_train = x[int(len(x)*size):]

    y_train = y[int(len(x)*size):]

    

    return x_train, y_train, x_val, y_val
x, y = randomDataset(1000, 2, random_state=0)
x_train, y_train, x_val, y_val = train_test_split(x, y, size=0.2, random_state=0)
l = LogisticsRegression()

learning_rate = 0.08

epoch = 250

l.train(x_train, y_train, learning_rate, epoch, random_state=0)

l.plot_convergence()

l.weight, l.bias
y_hat = l.predict(x_val)
confusion_matrix(y_val, y_hat)
df = pd.read_csv('../input/iris-dataset/Iris_binary.csv')

df.head(2)
df.Species.unique()
df.Species.replace(('Iris-setosa', 'Iris-versicolor'), (0, 1), inplace=True)
df = df.sample(frac=1, random_state=0)
X, Y = df.drop(['Species'], axis=1).values, df.Species.values

Y = Y.reshape(-1, 1)
X_train, Y_train, X_val, Y_val = train_test_split(X, Y, size=0.2, random_state=0)
l = LogisticsRegression()

learning_rate = 0.01

epoch = 100

l.train(X_train, Y_train, learning_rate, epoch, random_state=0)

l.plot_convergence()
Y_hat = l.predict(X_val)



confusion_matrix(Y_val, Y_hat)
print(classification_report(Y_val, Y_hat))
class SoftmaxRegression:

    

    def softmax(self, x):

        

        # shape(#samples, #classes)

        z = ((np.dot(x, self.weight)) + self.bias)

        

        # shape(#samples, #classes)

        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

        

    def forward(self, x):

        

        # shape(#samples, #classes)

        return self.softmax(x)

    

    def crossEntropy(self, y, y_hat):

        

        # shape(#samples, )

        return - np.sum(np.log(y_hat) * (y), axis=1)

    

    def cost(self, y, y_hat):

        

        # scalar

        return np.mean(self.crossEntropy(y, y_hat))

        

    def train(self, x, y, alpha, epoch, random_state=-1):

        

        # x : shape(#samples, #features)

        # y : shape(#samples, #classes)

        

        m, n, c = x.shape[0], x.shape[1], y.shape[1]

        

        if random_state != -1:

            np.random.seed(random_state)

        

        # shape(#features, #classes)

        self.weight = np.random.randn(n,c)



        # shape(1, #classes)

        self.bias = np.zeros((1,c))

        

        self.epoch = epoch

        

        self.cost_list = []

        

        for i in range(self.epoch):

            

            # shape(#samples, #classes)

            y_hat = self.forward(x)

    

            # scalar

            loss = self.cost(y, y_hat)



            self.cost_list.append(loss)



            # Gradient

            # dL_dw : dLoss/dweight (#features, #classes)

            dL_dw = (np.dot(x.T, (y_hat - y)))/m



            # dL_db : dLoss/dbias (1, #classes)

            dL_db = np.sum((y_hat - y)/m)



            # shape(#features, #classes)

            self.weight = self.weight - (alpha * dL_dw)



            # shape(1, #classes)

            self.bias = self.bias - (alpha * dL_db)

            

    def plot_convergence(self):

        

        plt.plot([i for i in range(self.epoch)], self.cost_list)

        plt.xlabel('Epochs'); plt.ylabel('Cross Entropy')

        

    def predict(self, x_test):

        

        # shape(#samples, #classes)

        y_hat = self.forward(x_test)

        return y_hat.argmax(axis=1)
df = pd.read_csv('../input/iris-dataset/Iris.csv')

df.head(2)
df.Species.unique()
df.Species.replace(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), (0, 1, 2), inplace=True)
df = df.sample(frac=1, random_state=0)
X, Y = df.drop(['Species'], axis=1).values, df.Species.values
X_train, Y_train, X_val, Y_val = train_test_split(X, Y, size=0.2, random_state=0)

Y_train = (np.arange(np.max(Y_train) + 1) == Y_train[:, None]).astype(float)
s = SoftmaxRegression()



s.train(X_train, Y_train, 0.02, 200, random_state=0)

s.plot_convergence()
Y_hat = s.predict(X_val)



confusion_matrix(Y_val, Y_hat)
print(classification_report(Y_val, Y_hat))