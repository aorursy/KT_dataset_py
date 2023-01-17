import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import norm

from scipy.stats import multivariate_normal as mvn
# Before starting its better to have concept of oops

# I will be commenting as much as I can for oops concept

# Start by creating class

# Refer to Bayes' Theorem, conditional probablity with Gaussian distribution (probablity density function)

class GaussianNaiveBayes:

    

    # fit methods trains the data

    def fit(self,X,y,spar=10e-3): # here self is the variable which refers to current object of class 

        number_of_sample,number_of_features = X.shape # returns shape of X which is NxD dimensional

        # categories contains classes in Y uniquely due to Set

        self.categories=np.unique(y)

        

        # number_of_classes is the local variable

        number_of_classes=len(self.categories)

        

        # Initialising mean, var and priors

        self.gaussian_mean=np.zeros((number_of_classes,number_of_features),dtype=np.float64)

        self.gaussian_var=np.zeros((number_of_classes,number_of_features),dtype=np.float64)

        self.log_prior=np.zeros((number_of_classes),dtype=np.float64)

        

        # Calculating mean,var,prior based on categories in Y

        for classes in self.categories:

            X_classes=X[classes==y] # grouping into X_classes array according to category in y

            self.gaussian_mean[classes,:]=X_classes.mean(axis=0) # mean with each row of sample belonging particular column(features)

            self.gaussian_var[classes,:]=X_classes.var(axis=0)+spar

            self.log_prior[classes]=np.log(X_classes.shape[0]/float(number_of_sample)) #number of sample in a class/ total samples

            # i have logged prior because in posterior we will be calculation log_pdf in predict

            

        

        

        

    

    

    # predict method make prediction

    def predict(self,X):

        # posterior probablity dimension (number of sample,number of categories)

        posteriorS=np.zeros((X.shape[0],len(self.categories)))

        for classes in self.categories: # calculating posterior with log of class_conditional probablity + log prior 

            posteriorS[:,classes]=mvn.logpdf(X,

                                             mean=self.gaussian_mean[classes,:],

                                             cov=self.gaussian_var[classes,:]) + self.log_prior[classes]

        return np.argmax(posteriorS,axis=1)

        

    def accuracy(self,y_true,predicted):

        return np.mean(y_true==predicted)

        

            

        
from sklearn import datasets

X,y=datasets.make_classification(n_samples=3000,n_features=10,n_classes=2,random_state=42)
# Splitting data into train and test set

# can use sklearn's train-test split but i want to keep it simple

X_train=X[:2900,:]

X_test=X[2900:,]

y_train=y[:2900]

y_test=y[2900:]
# Training data

nb=GaussianNaiveBayes()

nb.fit(X_train,y_train)
# predicting on test-set

pr=nb.predict(X_test)
# Checking Accuracy

nb.accuracy(y_test,pr)