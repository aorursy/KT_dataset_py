import numpy as np

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')
class LinearRegression:

        

    def forward(self, x):

    

        # shape(#samples, 1)

        return ((np.dot(x, self.weight)) + self.bias)

    

    def leastSquare(self, y, y_hat):

        

        # shape(#samples, 1)

        return (y_hat - y)**2

    

    def cost(self, y, y_hat):

        

        m = y.shape[0]

        

        # scalar

        return (np.sum(self.leastSquare(y, y_hat)))/(2*m)

        

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

        

        for i in range(epoch):

            

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

        plt.xlabel('Epochs'); plt.ylabel('Mean Squared Error')

        

    def predict(self, x_test):

        

        # shape(#samples, 1)

        return self.forward(x_test)
def randomDataset(m, n, random_state=-1):

    

    if random_state != -1:

        np.random.seed(random_state)

        

    x = np.random.randn(m, n)

    slope = np.random.randn(n, 1)

    epsilon = np.random.randn(1, 1)

    y = np.dot(x, slope) + epsilon

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
import numpy as np



def rss(y, y_hat):

    

    return np.sum((y-y_hat)**2)



def tss(y):

    

    return np.sum((y-y.mean())**2)



def r2(y, y_hat):

    

    return (1 - (rss(y, y_hat)/tss(y)))



def rmse(y, y_hat):

    

    return np.sqrt(np.mean((y-y_hat)**2))
m = 1000

n = 2

x, y = randomDataset(m, n, random_state=0)
x_train, y_train, x_val, y_val = train_test_split(x, y, size=0.2, random_state=0)
l = LinearRegression()
alpha = 0.1

epoch = 20

l.train(x_train, y_train, alpha, epoch, random_state=0)

l.plot_convergence()

l.weight, l.bias
y_hat = l.predict(x_val)

print(rmse(y_val, y_hat), r2(y_val, y_hat))