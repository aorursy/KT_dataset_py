import numpy as np

import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as stats

import math

from scipy.stats import multivariate_normal
mean = [0, 0]

cov = [[1, 0], [0, 1]]  # diagonal covariance



x, y = np.random.multivariate_normal(mean, cov, 10000).T

plt.plot(x, y, 'x')

plt.axis('equal')

plt.show()



plt.figure(num=None, figsize=(14, 12), dpi=80, facecolor='w', edgecolor='k')



r = np.stack([x, y], axis=1)

print(r.shape)



r_norm = np.linalg.norm(r, axis=1)

counts, bins = np.histogram(r_norm, bins=100, normed=True)



def _r_pdf(x):

    if bins[0] < x < bins[-1]:

        i = np.digitize(x, bins) - 1

        return counts[i]

    if x < bins[0]:

        return counts[0]

    return 1e-308

r_pdf = np.vectorize(_r_pdf)



x = np.linspace(0.0, 5.0, 10000)

plt.plot(x, r_pdf(x))



x = np.asarray(np.linspace(0.0, 5.0, 10000))

x = np.stack([x, np.zeros(10000)], axis=1)

var = multivariate_normal(mean, cov)

plt.plot(x, var.pdf(x))



def func(x):

    n = 2

    return math.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** n) * _r_pdf(x) 

func = np.vectorize(func)

    

plt.ylim(0, 0.9)



x = np.asarray(np.linspace(0.01, 5.0, 10000))

plt.plot(x, func(x))



plt.show()



mean = [0, 0]

cov = [[1, 0], [0, 1]]  # diagonal covariance



x, y = np.random.multivariate_normal(mean, cov, 10000).T

plt.plot(x, y, 'x')

plt.axis('equal')

plt.show()



plt.figure(num=None, figsize=(14, 12), dpi=80, facecolor='w', edgecolor='k')



r = np.stack([x, y], axis=1)

print(r.shape)



r_norm = np.linalg.norm(r, axis=1)

counts, bins = np.histogram(r_norm, bins=100, normed=True)



def _r_pdf(x):

    if bins[0] < x < bins[-1]:

        i = np.digitize(x, bins) - 1

        return counts[i]

    if x < bins[0]:

        return counts[0]

    return 1e-308

r_pdf = np.vectorize(_r_pdf)



x = np.linspace(0.0, 5.0, 10000)

plt.plot(x, r_pdf(x))



x = np.asarray(np.linspace(0.0, 5.0, 10000))

x = np.stack([x, np.zeros(10000)], axis=1)

var = multivariate_normal(mean, cov)

plt.plot(x, var.pdf(x))



def func(x):

    n = 2

    return math.gamma(n / 2.0) / (2.0 * np.pi ** (n / 2.0) * x ** (n-1)) * _r_pdf(x) 

func = np.vectorize(func)

    

plt.ylim(0, 0.9)



x = np.asarray(np.linspace(0.01, 5.0, 10000))

plt.plot(x, func(x))



plt.show()