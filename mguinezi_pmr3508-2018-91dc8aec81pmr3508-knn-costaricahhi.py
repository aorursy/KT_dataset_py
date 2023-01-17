# Data manipulation
import pandas as pd
import numpy as np
import sklearn
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Set a few plotting defaults
%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 18
plt.rcParams['patch.edgecolor'] = 'k'


HHItrain = pd.read_csv("../input/train.csv",sep=r'\s*,\s*', engine='python', na_values="?")
HHItest = pd.read_csv("../input/test.csv", sep=r'\s*,\s*',engine='python',na_values="?")
HHItrain.shape
HHItest.shape
HHItrain.head()
HHItest.head()
XHHItrain = HHItrain[["refrig","escolari","computer","television","rooms","hhsize","paredblolad","mobilephone","qmobilephone","overcrowding"]]
YHHItrain = HHItrain.Target
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, XHHItrain,YHHItrain,cv=10)
scores
knn.fit(XHHItrain,YHHItrain)
XHHItest = HHItest[["refrig","escolari","computer","television","rooms","hhsize","paredblolad","mobilephone","qmobilephone","overcrowding"]]
YHHItestPred = knn.predict(XHHItest)
YHHItestPred
arr1= HHItest.iloc[:,0].values
arr1 = arr1.ravel()
dataset = pd.DataFrame({'Id':arr1[:],'Target':YHHItestPred[:]})
dataset.to_csv("submition.csv", index = False)
# YHHItest = HHItest.Target

# from sklearn.metrics import accuracy_score
# Finalaccuracy = accuracy_score(YHHItest,YHHItestPred)
# Finalaccuracy
