# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

from sklearn.datasets import make_regression # For generating Linear regression dataset

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X, y = make_regression(n_samples=100, n_features=1, noise=50, bias=10, random_state=666)
X = X.reshape(-1)

y = y.reshape(-1)
plt.scatter(X, y)

plt.show()
# CONSTANTS

LEARNING_RATE = 0.1 # alpha

TOLERANCE = 0.000001 # Convergence criteria

N = X.shape[0]
# Initialize weights with random number or 0.

theta0, theta1 = 0, 0
def MSE(y, y_bar):

    return np.sum((y - y_bar) ** 2) / N



def predict(X, theta0, theta1):

    return theta0 + theta1 * X
E_prev = 0

epochs = 0

while True:

    y_bar = predict(X, theta0, theta1)

    E = MSE(y, y_bar)

    theta0 = theta0 - LEARNING_RATE * ((-2/N) * sum(y - y_bar))

    theta1 = theta1 - LEARNING_RATE * ((-2/N) * sum(X * (y - y_bar)))

    if abs(E - E_prev) <= TOLERANCE:

        break

    E_prev = E

    epochs += 1

    print(f'MSE after {epochs} epochs: {E}')
print(f'Learned parameters: theta0 = {theta0} and theta1 = {theta1}')
# Making predictions

Y_pred = predict(X, theta0, theta1)



plt.scatter(X, y)

plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # plotting regression line

plt.show()