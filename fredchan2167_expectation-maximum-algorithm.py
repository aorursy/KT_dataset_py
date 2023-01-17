# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from sklearn.datasets import make_spd_matrix

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
np.random.seed(5)

n = 100

mu1 = np.array([3,1]).T

mu2 = np.array([-2,2]).T



std1 = np.array([[0.5,1],

                 [1,0.5]])



std2 = np.array([[0.5,-1],

                 [-1,0.5]])



cluster1 = np.random.randn(n,2) @ std1 + mu1

cluster2 = np.random.randn(n,2) @ std2 + mu2

X = np.concatenate((cluster1, cluster2), axis=0)

index = np.random.randint(X.shape[0], size = 5)

Xte_n = np.array([[0,5],

                [-4,-1],

                 [4,-2]])

Xte_p = X[index,:]
'''

Xte: test point that are neither of the class, looking for low probability

Xte: test point that are within gassuian distribution 

'''



plt.plot(cluster1[:,0], cluster1[:,1], '.',label='cluster1')

plt.plot(cluster2[:,0], cluster2[:,1], '.',label='cluster2')

plt.plot(Xte_n[:,0], Xte_n[:,1], 'o', label='negative test point')

plt.plot(Xte_p[:,0], Xte_p[:,1], 'o', label='positive test point')

plt.legend()

plt.show()
def K_mean (X, k, iteration= 5, verbose= False):

    '''

    Input: 

    X = data with shape (m, n)

    k = # of clusters

    

    Return:

    labels = appointed label taken the value from 0....k-1

             w/ shape (m,)

    '''



    # return k random number of index 

    index = np.random.randint(X.shape[0], size=k)

    

    # assign centroid randomly by index

    centroids = X[index,:]

    distances = np.zeros((X.shape[0],k))

    

    for i in range(iteration):

        

        #calculate distance from centroid

        for label in range(k):

            distances[:,label] = np.linalg.norm(X - centroids[label,:], axis=1)



        #generate labels from minimum distances

        labels = np.argmin(distances, axis=1)





        #update cluster position according labels

        clusters = []

        

        for label in range(k):

            ind = np.where(labels == label)

            cluster = np.squeeze(X[ind,:])

            centroids[label,:] = np.sum(cluster, axis=0) / len(cluster)

            clusters.append(cluster)

            

       

        

        if verbose:

            for cluster in clusters:

                plt.plot(cluster[:,0], cluster[:,1],'.')

            plt.plot(centroids[:,0], centroids[:,1],'^', label='centroid')

            plt.legend()

            plt.show()

        

           

    

    return labels, clusters, centroids

        
class Expectation_maximum:

    

    def __init__(self, Xtr):

        self.X = Xtr

        

    def E_step (self, mu, sigmas, prior):

        '''

        Note:

            multivariate_normal.pdf = variable mean only take shape (n,)

            

   

            X = data with shape (m, n)

            k = number of gaussians

            mu = (k,n)

            Sigma = is a list of k number of (n,n) matrix

            prior = (1,k)

            W = (m, k)

        '''



        m, n = self.X.shape

        P_X = np.zeros((m,self.k))

        num = np.zeros((m,self.k))





        for i in range(self.k):   

            P_X[:,i] = multivariate_normal.pdf(X,mean=mu[i,:], cov=sigmas[i],allow_singular=True) #(m,) P(X|Z = label; mu, sigmas)

            num[:,i] = P_X[:,i] * prior[:,i]



        W = num / np.sum(num, axis=1, keepdims=True)

        

        return W

        

    def M_step (self, W):

        '''

        X = (m, n)

        k = real number of gaussians

        mu = (k,n)

        Sigma = a list of k number of (n,n) matrix

        prior = (1,k)

        W = (m, k)

        '''

        eps = 1e-8

        m, n = self.X.shape

        prior = np.zeros((1,self.k))

        mu = np.zeros((self.k, n))

        sigmas = []

        for i in range(self.k):

            prior[:,i] = np.sum(W[:,i], axis=0, keepdims=True) / m

            mu[i,:] = np.sum(W[:,i].reshape(-1,1) * self.X, axis=0, keepdims=True) / np.sum(W[:,i], axis=0, keepdims=True)

            sigma = ((W[:,i].reshape(-1,1) * (self.X - mu[i,:])).T @ (self.X - mu[i,:]) + eps) / (np.sum(W[:,0]) + eps)

            sigmas.append(sigma)

            

        return mu, sigmas, prior





    def train (self,k, iteration=50):

        

        '''

        k: number of gaussian distribution

        

        Note:

            np.cov: take (n,m) matrix,

                    bias = Ture means devide by n, instead n-1 (default)

        '''

        self.k = k

        m, n = self.X.shape

        mu = np.zeros((self.k, n))

        prior = np.zeros((1,k))

        sigmas = []

        

        labels, clusters, _ = K_mean(self.X, k = k)

        

         # initial parameters using k-mean result

        for i in range(k):

            

            mu[i,:] = np.mean(clusters[i], axis=0)

            

            sigma = np.cov(clusters[i].T, bias=True)

            sigmas.append(sigma)

            

            ind = np.where(labels == i)

            prior[:,i] = len(ind) / m

            



        for i in range(iteration):

            W = self.E_step(mu, sigmas, prior)

            mu, sigmas, prior = self.M_step(W)

        

        self.W = W

        self.mu = mu

        self.sigmas = sigmas

        self.prior = prior

    

    def predict(self, Xte):

        

        m, n = Xte.shape

        p_X = np.zeros((m, self.k))

        num = np.zeros((m, self.k))

        for i in range(self.k):   

            p_X[:,i] = multivariate_normal.pdf(Xte,mean=self.mu[i,:], cov=self.sigmas[i], allow_singular=True) #(m,) P(X|Z = label; mu, sigmas)

            num[:,i] = p_X[:,i] * self.prior[:,i]

            

        result = np.sum(num, axis=1).reshape(-1,1)

        

        

        return result
model = Expectation_maximum(X)

model.train(k = 2, iteration=100)

prediction = model.predict(Xte_p)





prediction