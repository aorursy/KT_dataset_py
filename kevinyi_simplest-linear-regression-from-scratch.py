import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set()

rnd = np.random.RandomState(10)
X = 10 * rnd.rand(45)
Y = 3 * X + 3 + rnd.rand(45)
plt.scatter(X, Y, s=2)
x_mean = np.mean(X)
y_mean = np.mean(Y)

n = len(X)

numerator = 0
denominator = 0
for i in range(n):
    numerator += (X[i] - x_mean)*(Y[i] - y_mean)
    denominator += (X[i] - x_mean)**2

w = numerator / denominator
b = y_mean - (x_mean*w)
plt.scatter(X, Y, s=2)
plt.plot(X, w*X + b, 'r',linewidth=1)