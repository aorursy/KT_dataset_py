import numpy as np

import matplotlib.pyplot as plt



class LogisticRegression:

    def __init__(self):

        self._theta = None

        self._J_history = None



    def fit(self, X, y, alpha=0.01, iter_num=1200, epsion=1e-4):

        X = np.c_[np.ones(X.shape[0]), X]

        y = np.c_[y]

        theta = np.zeros((X.shape[1],1))

        iter_n = 0

        m = X.shape[0]



        def sigmoid(z):

            return 1.0 / (1 + np.exp(-z))



        def costFun(X, theta, y):

            m = X.shape[0]

            return (1 / m) * np.dot(y.T, np.log(X.dot(theta)) + np.dot((1 - y).T, np.log(X.dot(theta))))



        while iter_n < iter_num:

            hx = sigmoid(X.dot(theta)).reshape(-1,1)

            last_theta = theta

            theta -= (alpha / m) * X.T.dot(hx - y)

            iter_n += 1

        self._theta = theta

    def score(self,X,y):

        X = np.c_[np.ones(X.shape[0]), X]

        y = np.c_[y]

        m = X.shape[0]

        mse = (1/m)*np.sum(np.square(self.sigmoid(X.dot(self._theta))- y))

        var = (1/m)*np.sum(np.square(y - y.mean()))

        return 1 - mse/var

    def sigmoid(self,z):

        return 1.0 / (1 + np.exp(-z))

    def predict_proba(self,X):

        X = np.c_[np.ones(X.shape[0]), X]

        res = []

        for i in self.sigmoid(X.dot(self._theta))>0.5:

            if i[0]:

                res.append(1)

            else:

                res.append(0)

        return np.array(res)



    def showLogis(self,X, y):

        X = np.c_[np.ones(X.shape[0]), X]

        y = np.c_[y]

        m, f = X.shape

        plt.figure('Line diagram')

        plt.scatter(X[y[:,0]==1,1],X[y[:,0]==1,2],c='b',label='good')

        plt.scatter(X[y[:,0]==0,1],X[y[:,0]==0,2],c='r',label='bad')

        plt.legend(loc='best')

        # 画分界线

        min_x = min(X[:, 1])  # The minimum of x1

        max_x = max(X[:, 2])  # The maximum value of x1

        y_min_x = (-self._theta[0] - self._theta[1] * min_x) / self._theta[2]

        #The minimum value of x1 corresponds to the value of x2

        y_max_x = (-self._theta[0] - self._theta[1] * max_x) / self._theta[2]

        # The maximum value of x1 is x2

        plt.plot([min_x, max_x], [y_min_x, y_max_x], '-g')  # Painting line

        plt.show()



    def testAccuracy(self,X, y):

        X = np.c_[np.ones(X.shape[0]), X]

        y = np.c_[y]

        m = X.shape[0]

        count = 0



        for i in range(m):

            h = self.sigmoid(np.dot(X[i, :], self._theta))  # Calculate the predicted value

            if bool(np.where(h >= 0.5, 1, 0)) == bool(y[i]):

                count += 1

        return count / m

    def accuracy(self,X, y):

        m = X.shape[0]  # 样本个数

        count = 0

        y_pred = self.predict_proba(X)

        return np.sum(y == y_pred)/len(y)
import numpy as np

from sklearn.datasets import make_moons

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib as mpl





mpl.rcParams['axes.unicode_minus'] = False

h = .02

x, y = make_moons(250,noise=0.25)

from matplotlib.colors import ListedColormap

# Drawing board

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# standardized

x= StandardScaler().fit_transform(x)

# Cut into training and test sets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Find the maximum and minimum values for each axis

x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5

y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5

# Grid point

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

# Into the model

clf = LogisticRegression()

# Training data

clf.fit(X_train, y_train)

# To calculate R2

score = clf.score(X_test, y_test)

Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

cm = plt.cm.RdBu

plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)



# Plot sample points

cm_bright = ListedColormap(['#FF0000', '#0000FF'])

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,

               edgecolors='k',label="The training set")

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.5,

               edgecolors='k',label="The test set")

plt.title("Logistic regression classification")

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.xticks()

plt.yticks()

plt.legend()

# put R2

plt.text(xx.max() - .3, yy.min() + .3, ('%  s%.2f' % ("%",score*100)),size=15, horizontalalignment='right')

plt.show()

# Display accuracy

print(clf.accuracy(x,y))