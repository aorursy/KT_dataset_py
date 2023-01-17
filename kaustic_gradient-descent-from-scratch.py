# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.random.seed(42)
# 60 = 21 * m + c

# m_new = (y_actual - y_pred) / x = (60 - y_pred) / 21

# c_new = (y_actual - y_pred)



y_actual = 60

x = 21

n_iterations = 10

eta = 0.2



m = np.random.randn()

c = np.random.randn()

for i in range(n_iterations):

    y_pred = x * m + c

    diff = y_actual - y_pred

    print(f"Iteration: {i}:\n  Prediction: {y_pred}, Error: {diff}")

    m += (diff / x) * eta

    c += diff * eta
m, c
X = 2 * np.random.rand(100, 1)

y = 4 + 3 * X + np.random.randn(100, 1)
import matplotlib.pyplot as plt
plt.style.use("seaborn-darkgrid")



plt.plot(X, y, "b.")

plt.xlabel("$X_1$", fontsize=18)

plt.ylabel("$y$", rotation=0, fontsize=18)

plt.axis([0, 2, 0, 15])

plt.show()
# Data params

n_samples = X.shape[0]

n_features = X.shape[1]



# Hyperparameters

n_epochs = 100

learning_rate = 0.25





# Initialization

weights = np.random.randn(n_features)

bias = np.random.randn()



for epoch in range(1, n_epochs + 1):

    # Linear equation -> Y = weights . X + bias

    # Partial differentiation wrt weights -> delta(Y)/delta(weights) = X

    # => weights_new = weights_old - (delta(Y) / X) * learning_rate

    

    y_pred = X * weights + bias

    diff = y_pred - y

    weights -= diff.mean(axis=0) * learning_rate

    bias -= diff.mean() * learning_rate

    rmse = np.sqrt(np.square(diff).mean())

    print(f"Epoch: {epoch}\nRMSE: {rmse}\nDiff mean: {diff.mean()}\n")
X_new = np.array([[0], [2]], dtype=np.float64)

y_pred_new = X_new * weights + bias

y_pred_new
plt.style.use("seaborn-darkgrid")



plt.plot(X_new, y_pred_new, "g-")

plt.plot(X, y, "b.")

plt.axis([0, 2, 0, 15])

plt.show()
# Let's try with median instead of mean



# Data params

n_samples = X.shape[0]

n_features = X.shape[1]



# Hyperparameters

n_epochs = 100

learning_rate = 0.25





# Initialization

weights = np.random.randn(n_features)

bias = np.random.randn()



for epoch in range(1, n_epochs + 1):

    # Linear equation -> Y = weights . X + bias

    # Partial differentiation wrt weights -> delta(Y)/delta(weights) = X

    # => weights_new = weights_old - (delta(Y) / X) * learning_rate

    

    y_pred = X * weights + bias

    diff = y_pred - y

    weights -= np.median(diff, axis=0) * learning_rate

    bias -= np.median(diff) * learning_rate

    rmse = np.sqrt(np.square(diff).mean())

    print(f"Epoch: {epoch}\nRMSE: {rmse}\nDiff mean: {diff.mean()}\n")
X_new = np.array([[0], [2]], dtype=np.float64)

y_pred_new = X_new * weights + bias

print(y_pred_new)



plt.style.use("seaborn-darkgrid")



plt.plot(X_new, y_pred_new, "g-")

plt.plot(X, y, "b.")

plt.axis([0, 2, 0, 15])

plt.show()
# Standard approach



# Data params

n_samples = X.shape[0]

n_features = X.shape[1]



# Hyperparameters

n_epochs = 100

learning_rate = 0.25





# Initialization

weights = np.random.randn(n_features)

bias = np.random.randn()



for epoch in range(1, n_epochs + 1):

    # Linear equation -> Y = weights . X + bias

    # Partial differentiation wrt weights -> delta(Y)/delta(weights) = X

    # => weights_new = weights_old - (delta(Y) / X) * learning_rate

    

    y_pred = X * weights + bias

    diff = y_pred - y



    weights -= (X.T.dot(diff).mean(axis=0) / n_samples) * learning_rate

    bias -= diff.mean() * learning_rate

    rmse = np.sqrt(np.square(diff).mean())

    print(f"Epoch: {epoch}\nRMSE: {rmse}\nDiff mean: {diff.mean()}\n")
X_new = np.array([[0], [2]], dtype=np.float64)

y_pred_new = X_new * weights + bias

print(y_pred_new)



plt.style.use("seaborn-darkgrid")



plt.plot(X_new, y_pred_new, "g-")

plt.plot(X, y, "b.")

plt.axis([0, 2, 0, 15])

plt.show()