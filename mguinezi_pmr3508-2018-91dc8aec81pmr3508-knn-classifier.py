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
XHHItrain = HHItrain[["refrig","escolari","computer","television"]]
YHHItrain = HHItrain.Target
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, XHHItrain,YHHItrain,cv=10)
scores
knn.fit(XHHItrain,YHHItrain)
XHHItest = HHItest[["refrig","escolari","computer","television"]]
YHHItestPred = knn.predict(XHHItest)
YHHItestPred
# YHHItest = HHItest.Target

# from sklearn.metrics import accuracy_score
# Finalaccuracy = accuracy_score(YHHItest,YHHItestPred)
# Finalaccuracy