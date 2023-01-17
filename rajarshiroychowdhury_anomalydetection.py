import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

import scipy.io
def estimateGaussian(X):

    

    (m, n) = np.shape(X)

    mu = np.zeros(n,dtype=np.double)

    sigma2 = np.zeros(n, dtype=np.double)

    mu = np.sum(X,0)/m;

    xnew = np.power(np.subtract(X, mu), 2)

    sigma2 = np.sum(xnew,0)/m

    return mu, sigma2
def multivariateGaussian(X, mu, Sigma2):

    

    k = np.size(mu,0)

    X = np.subtract(X, np.transpose(mu))

    Sigma2 = np.diag(Sigma2)    

    A = np.sum(np.multiply(np.dot(X, np.linalg.pinv(Sigma2)), X),1)

    p = (2*np.pi)**(-k/2) * np.linalg.det(Sigma2)**(-0.5) * np.exp(np.dot(-0.5,A))

    return p
def visualizeFit(X, mu, sigma2, outliers):

    

    x = np.linspace(0, 35, 71)

    y = np.linspace(0, 35, 71)

    X1, X2 = np.meshgrid(x, y)

    X1S = np.reshape(X1,np.size(X1,0)*np.size(X1,1), order='F')

    X2S = np.reshape(X2,np.size(X2,0)*np.size(X2,1), order='F')

    Z = multivariateGaussian(np.column_stack((X1S,X2S)), mu, sigma2)

    Z = np.reshape(Z, np.shape(X1), order='F')

    levels_r = np.array([10**(-20), 10**(-17), 10**(-14), 10**(-11), 10**(-8), 10**(-5), 10**(-2)])

    levels = levels_r.reshape((levels_r.shape[0],1))



    fig= plt.figure(figsize=(12,9))

    axes= fig.add_axes([0.1, 0.1, 0.8, 0.8])

    axes.plot(X[:,0], X[:,1], 'gx', markersize=6)

    axes.contour(X1, X2, Z, levels[:,0])

    axes.plot(X[outliers[:], 0], X[outliers[:], 1], 'ro', mfc='none', markersize=12)

    axes.set_xlim([0, 30])

    axes.set_ylim([0, 30])

    fig.suptitle('Anomaly Detection : Probability Contours and Outliers', fontsize=22)

    axes.set_xlabel('Latency (ms)', fontsize=15)

    axes.set_ylabel('Thoroughput (mb/s)', fontsize=15)

    plt.show()
def selectThreshold(yval, pval):

    

    bestepsilon = 0

    bestF1 = 0

    F1 = 0

    stepsize = (np.amax(pval) - np.amin(pval))/1000

    epsilon = np.amin(pval)

    temp = np.zeros(np.shape(yval))

    while epsilon < np.amax(pval):

        predictions = (pval < epsilon).astype(int)

        for i in range (0,307):

            temp[i] = yval[i,0]*predictions[i]

        precision = np.sum(temp)/np.sum(predictions)

        recall    = np.sum(temp)/np.sum(yval)

        F1 = 2*precision*recall/(precision+recall)        

        if F1 > bestF1:

            bestF1 = F1

            bestepsilon = epsilon

        epsilon += stepsize

    return bestepsilon, bestF1
data = scipy.io.loadmat('../input/ex8data1.mat')

X = data['X']

Xval = data['Xval']

yval = data['yval']

plt.plot(X[:,0], X[:,1], 'bx')

plt.ylim((0,30))

plt.xlim((0,30))

plt.xlabel('Latency (ms)')

plt.ylabel('Throughput (mb/s)')

plt.title('DATASET')

plt.show()
mu, sigma2 = estimateGaussian(X)

p = multivariateGaussian(X, mu, sigma2)

pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval, pval)

outliers = np.asarray(np.nonzero(p < epsilon)).T

visualizeFit(X, mu, sigma2, outliers)