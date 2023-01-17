# Like, the 'Protected Division' usually returns '1' on having a denominator of 0, whereas the paper on which the Paper of 'Improving fitness...' is based on mentions the protected division to be 0 if the denominator is 0
import numpy as np

import matplotlib

import sys

import pandas as pd

import time

import math

import matplotlib.pyplot as plt

import random

import tpot as tp

from sklearn.cross_validation import train_test_split

from gplearn import genetic

import gplearn as gp

import itertools

%matplotlib inline
dataSet = pd.read_csv("../input/creditcard.csv")

dataSet.head()



# 30 features with Time and Amount + 28 features from PCA
# Checking the Counts of different classes

# The positive cases for our tests are 492 frauds compared to 284315 for the genuine transactions

dataSet['Class'].value_counts()
# Population Size

pSize = 500



# Number of Generations

numGen = 1000



# Crossover Probability

pCross = 0.9



# Mutation Probability

pMut = 0.1



# Tournament Size

tSize = 3
# Creating a DataFrame to store the normalized values of the orignal data with same column names

normDataSet = pd.DataFrame(columns=dataSet.columns)
# Normalizing the data using min-max normalization

# normalized value = (value - min(attribute))/(max(attribute)-min(attribute))

for i in range(30):

   normDataSet[normDataSet.columns[i]]=(dataSet[dataSet.columns[i]]- min(dataSet[dataSet.columns[i]]))/(max(dataSet[dataSet.columns[i]])-min(dataSet[dataSet.columns[i]]))
normDataSet['Class']=dataSet['Class']

normDataSet
# Defining Conditional if

# if first argument is negative, return the second, otherwise third argument

# This is done to have an additional operator and avoiding having only smooth decision boundaries

def cond_if(arg1, arg2, arg3):

    return np.where(arg1<0,arg2,arg3)

    

# The function for protected division which returns 0 if the denominator is 0

def p_div(x1,x2):

    with np.errstate(divide='ignore',invalid='ignore'):

        return np.where(np.abs(x2)>0.001, np.divide(x1,x2),0.)
# Including the conditional if in the make functions of gp learn

cif = gp.functions.make_function(function=cond_if, name = 'cif', arity=3)

pdiv = gp.functions.make_function(function=p_div,name='pdiv',arity=2)
# Creating the new Errors mean fitness function

# f_errors_mean = (TP)/(TP+FN) + TN/(TN+FP) + (1-Err_mean_min)+(1-Err_mean_maj)

# TP = True Positive, TN = True Negative, FP = False Positive, FN = False Negative, Err_mean_min = Mean Error Minority class, Err_mean_maj = Mean Error Majority Class

def _fem(y,y_pred,w):

    TP = len([x for x,z in zip(y_pred,y) if ((x<=0) and (z==0))])

    TN = len([x for x,z in zip(y_pred,y) if ((x>0) and (z==1))])

    FP = len([x for x,z in zip(y_pred,y) if ((x<=0) and (z==1))])

    FN = len([x for x,z in zip(y_pred,y) if ((x>0) and (z==0))])

    err_min = p_div(np.sum(np.abs(x) for x,z in zip(y_pred,y) if ((x>0) and (z==0))),FN)

    err_maj = p_div(np.sum(np.abs(x) for x,z in zip(y_pred,y) if ((x<=0) and (z==1))),FP)   

    f = p_div(TP,(TP + FN))+p_div(TN,(TN+FP))+(1-err_min)+(1-err_maj)

    return f

fem = gp.fitness.make_fitness(_fem,greater_is_better=True)
est_gp = genetic.SymbolicRegressor(metric=fem, population_size = pSize, generations = numGen, tournament_size = tSize, p_crossover = pCross, p_subtree_mutation=pMut, p_hoist_mutation=0.0,p_point_mutation=0.0,function_set=['add','sub','mul',pdiv,cif],verbose=1)
# Sampling 70% of the dataset randomly

tot_train = normDataSet.sample(frac = 0.7)

# Getting the gp-tree for the training data, first 30 columns as values and 31st as the Class Label

est_gp.fit(tot_train.iloc[:,0:30],tot_train['Class'])
# Work in progress, to be updated soon