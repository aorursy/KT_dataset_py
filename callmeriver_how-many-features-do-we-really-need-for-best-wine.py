# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv('../input/winequality-red.csv')

data.head()
# X = data[[data.columns]]
X = data.drop('quality',axis=1)
y = data.quality
X.head()
X = preprocessing.StandardScaler().fit(X).transform(X)
model = PCA()
results = model.fit(X)
plt.plot(results.explained_variance_)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

gnb = GaussianNB()
fit = gnb.fit(X,y)
pred = fit.predict(X)
print (confusion_matrix(pred,y))
print("accuracy: ")
print(confusion_matrix(pred,y).trace()*100/confusion_matrix(pred,y).sum())
predicted_correct = []
for i in range(1,10):
    model = PCA(n_components = i)
    results = model.fit(X)
    Z = results.transform(X)
    fit = gnb.fit(Z,y)
    pred = fit.predict(Z)
    predicted_correct.append(confusion_matrix(pred,y).trace())
plt.plot(predicted_correct)
plt.show()


