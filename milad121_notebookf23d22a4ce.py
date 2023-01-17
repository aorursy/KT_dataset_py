# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from sklearn import ensemble

from sklearn.utils import shuffle

import matplotlib.pyplot as plt

from sklearn import grid_search

from sklearn import cross_validation;

import xgboost as xgb

from sklearn import neighbors

from sklearn import cluster

from sklearn.metrics import pairwise

from sklearn.externals import joblib

from sklearn import feature_selection

from sklearn import decomposition







trainData= pd.read_csv("../input/train.csv");

correctClass=trainData['target'];

classes=np.unique(correctClass);

trainFeatures=trainData.drop(['id', 'target'], axis=1);





testData= pd.read_csv("../input/test.csv");

testFeatures=testData.drop(['id'], axis=1);

testIDs=testData['id'];



sumColumn=trainFeatures.apply(lambda row: row.sum(), axis=1);

nonZerosColumn=trainFeatures.apply(lambda row: sum(row!=0), axis=1);

trainFeatures['sum']=sumColumn

trainFeatures['nonZero']=nonZerosColumn;



sumColumnT=testFeatures.apply(lambda row: row.sum(), axis=1);

nonZerosColumnT=testFeatures.apply(lambda row: sum(row!=0), axis=1);

testFeatures['sum']=sumColumnT

testFeatures['nonZero']=nonZerosColumnT



df = pd.DataFrame(trainData)



# Any results you write to the current directory are saved as output.