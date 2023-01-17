import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import sklearn



def irispairplots(iris):



    flowers=iris.target_names



    features=iris.feature_names



    data = iris.data



    target = iris.target



    fig, axes = plt.subplots(4, 4, figsize = (15,15))

    for i in range(0,4):

        for j in range(0,4):

            if i == j or j==i:

                axes[i,j].hist(data[:,j],bins=20)

                axes[i,j].set_xlabel(features[j]) 

                axes[i,j].set_ylabel(features[i])

            else:

                axes[i,j].scatter(data[:,j],data[:,i],c=target)

                axes[i,j].set_xlabel(features[j]) 

                axes[i,j].set_ylabel(features[i])



from sklearn.datasets import load_iris

iris = sklearn.datasets.load_iris()

irispairplots(iris)

plt.show()