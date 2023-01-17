import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



df = pd.read_csv('../input/iris/Iris.csv')
# Heading of columns

df.columns
# This will return first 5 rows of data

df.head()
# This will return the information about columns such as counts and data type 

df.info()
# Very important function for getting statastical glimpse of data: such as mean, median, maximum and minimun entries and 25th, 50th and 75th percentile.

df.describe()
# adding first 100 rows of second and third columns which are SepalLengthCm, SepalWidthCm to X.

X = df.iloc[0:100, [1, 2]].values

# adding the classified species of first 100 rows to y.

y = df.iloc[0:100, 5].values
# np.where syntax goes like this : Where True, yield x, otherwise yield y.

# set output lable value to 1 if it is setosa and 0 if versicolor.

y = np.where(y == 'Iris-setosa', 1, 0)
X_std = np.copy(X)



# Standardization of variables = subtracting the mean and dividing by the standard deviation



X_std[:,0] = (X_std[:,0] - X_std[:,0].mean()) / X_std[:,0].std()

X_std[:,1] = (X_std[:,1] - X_std[:,1].mean()) / X_std[:,1].std()

# Define Logistic Regression hypothesis or sigmoid function

def sigmoid(X, theta):

    

    z = np.dot(X, theta[1:]) + theta[0]

    

    return 1.0 / ( 1.0 + np.exp(-z))
# Define Logistic Regression Cost Function

def lrCostFunction(y, hx):

  

    # compute cost for given theta parameters

    j = -y.dot(np.log(hx)) - ((1 - y).dot(np.log(1-hx)))

    

    return j
# Gradient Descent function to minimize the Logistic Regression Cost Function.

def lrGradient(X, y, theta, alpha, num_iter):

    # empty list to store the value of the cost function over number of iterations

    cost = []

    

    for i in range(num_iter):

        # call sigmoid function 

        hx = sigmoid(X, theta)

        # calculate error

        error = hx - y

        # calculate gradient

        # gradient is dot product of transpose of features(here X.T) and error(predictions(hx) - labels(y))

        grad = X.T.dot(error)

        # update values in theta

        theta[0] = theta[0] - alpha * error.sum()

        theta[1:] = theta[1:] - alpha * grad

        

        cost.append(lrCostFunction(y, hx))

        

    return cost        

# m = Number of training examples

# n = number of features

m, n = X.shape





# initialize theta(weights) parameters to zeros

theta = np.zeros(1+n)



# set learning rate to 0.01 and number of iterations to 500

alpha = 0.01

num_iter = 500
cost = lrGradient(X_std, y, theta, alpha, num_iter)
plt.plot(range(1, len(cost) + 1), cost)

plt.xlabel('Iterations')

plt.ylabel('Cost')

plt.title('Logistic Regression')
# print theta paramters 

print ('\n Logisitc Regression bias(intercept) term :', theta[0])

print ('\n Logisitc Regression estimated coefficients (SepalLenght, SepalWidth) :', theta[1:])
# function to predict the output label using the parameters

def lrPredict(X):

    

    # set output lable value to 1(setosa) if it is gt 0.5 and 0(versicolor) if not.

    return np.where(sigmoid(X,theta) >= 0.5, 1, 0)
y_pred = lrPredict(X_std)



# To check accuracy score

from sklearn.metrics import accuracy_score



print(accuracy_score(y, y_pred, normalize=True)*100)
from matplotlib.colors import ListedColormap



def plot_decision_boundry(X, y, classifier, h=0.02):

    # h = step size in the mesh

  

    # setup marker generator and color map

    markers = ('s', 'x', 'o', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])



    # plot the decision surface

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),

                         np.arange(x2_min, x2_max, h))

    Z = classifier(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.ylim(xx2.min(), xx2.max())



    # plot class samples

    for idx, cl in enumerate(np.unique(y)):

        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],

                    alpha=0.8, c=cmap(idx),

                    marker=markers[idx], label=cl)



plot_decision_boundry(X_std, y, classifier=lrPredict)

plt.title('Standardized Logistic Regression - Gradient Descent')

plt.xlabel('sepal length ')

plt.ylabel('sepal width ')

plt.legend(loc='upper left')

plt.tight_layout()
