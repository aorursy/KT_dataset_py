import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin



%matplotlib inline



plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (12,5)
class MySGDClassifier(BaseEstimator, ClassifierMixin):

    """

    Suggest that sample data contain only 2 classes

    """



    def __init__(self, C=1, alpha=0.01, max_epoch=10, penalty=None, fit_intercept=True, verbose=False):

        """

        C - коэф. регуляризации

        alpha - скорость спуска

        max_epoch - максимальное количество эпох

        """



        self.C = C

        self.alpha = alpha

        self.max_epoch = max_epoch

        self.penalty = penalty

        self.fit_intercept = fit_intercept

        self.verbose = verbose

        self.losses = []



    def __add_intercept(self, X):

        intercept = np.ones((X.shape[0], 1))

        return np.concatenate((intercept, X), axis=1)



    def __sigmoid(self, z):

        return 1 / (1 + np.exp(-z))



    def __loss(self, h, y):

        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()



    def fit(self, X, y=None):

        '''

        Train model

        '''

        if self.fit_intercept:

            X = self.__add_intercept(X)



        # weights initialization

        self.theta = np.zeros(X.shape[1])



        for i in range(self.max_epoch):

            z = np.dot(X, self.theta)

            h = self.__sigmoid(z)

            if not self.penalty:

                reg_cost = 0

            elif self.penalty == 'l2':

                reg_cost = 2 * self.theta / self.C

            elif self.penalty == 'l1':

                reg_cost = 1 / self.C

            else:

                raise Exception('Unknown method penalty')

            gradient = np.dot(X.T, (h - y)) / y.size + reg_cost

            self.theta -= self.alpha * gradient



            if(self.verbose == True and i % 10000 == 0):

                z = np.dot(X, self.theta)

                h = self.__sigmoid(z)

                print(f'loss: {self.__loss(h, y)} \t')

            self.losses.append(self.__loss(h, y))

        return self



    def partial_fit(self, X, y=None):

        return self



    def predict(self, X):

        '''

        Return class label

        '''

        y_hat = self.predict_proba(X).round()

        return y_hat



    def predict_proba(self, X):

        '''

        Return the probability of each class

        '''

        if self.fit_intercept:

            X = self.__add_intercept(X)



        y_hat_proba = self.__sigmoid(np.dot(X, self.theta))

        return y_hat_proba
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data[:, :2]

y = (iris.target != 0) * 1

model = MySGDClassifier(C=100, alpha=0.1, max_epoch=300000, penalty=None, verbose=True)

%time model.fit(X, y)



preds = model.predict(X)

(preds == y).mean()
np.random.seed(0)



C1 = np.array([[0., -0.8], [1.5, 0.8]])

C2 = np.array([[1., -0.7], [2., 0.7]])

gauss1 = np.dot(np.random.randn(200, 2) + np.array([5, 3]), C1)

gauss2 = np.dot(np.random.randn(200, 2) + np.array([1.5, 0]), C2)



X = np.vstack([gauss1, gauss2])

y = np.r_[np.ones(200), np.zeros(200)]



model = MySGDClassifier(C=100, alpha=0.1, max_epoch=300000, penalty='l2')

model.fit(X, y)



preds = model.predict(X)

# accuracy

print((preds == y).mean())

print(model.theta)



plt.scatter(X[:,0], X[:,1], c=y)



x1_min, x1_max = X[:,0].min(), X[:,0].max(),

x2_min, x2_max = X[:,1].min(), X[:,1].max(),

xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))

grid = np.c_[xx1.ravel(), xx2.ravel()]

probs = model.predict_proba(grid).reshape(xx1.shape)

plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='red')

plt.xlabel("X0 (feature)")

plt.ylabel("X1 (feature)")
alphas = [0.1, 0.01, 0.001]



fig, ax = plt.subplots()

for a in alphas:

    model = MySGDClassifier(C=100, alpha=a, max_epoch=100000, penalty='l2')

    model.fit(X, y)

    ax.plot(range(len(model.losses)), model.losses, label=str(a))



ax.set_title("Convergence analysis")

ax.set_xlabel("iteration")

ax.set_ylabel("alpha")

plt.legend()

plt.show()