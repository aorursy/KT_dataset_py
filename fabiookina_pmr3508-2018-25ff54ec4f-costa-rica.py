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
trainCR = pd.read_csv("../input/train.csv",na_values="?")
testCR = pd.read_csv("../input/test.csv",na_values="?")
trainCR.shape
testCR.shape
trainCR.head()
ntrainCR = trainCR.fillna(0)
ntestCR = testCR.fillna(0)
from sklearn import preprocessing
numtrainCR = ntrainCR.apply(preprocessing.LabelEncoder().fit_transform)
numtestCR = ntestCR.apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=100)
XtrainCR=numtrainCR.iloc[:,1:142]
YtrainCR=numtrainCR.Target
XtestCR=numtestCR.iloc[:,1:142]

scores = cross_val_score(knn, XtrainCR, YtrainCR, cv=20)
scores
knn.fit(XtrainCR,YtrainCR)
YtestPred = knn.predict(XtestCR)
ntestCR["Id"]
ID=ntestCR["Id"]
result=np.column_stack((ID,YtestPred))
x = ["Id","Target"]
Prediction = pd.DataFrame(columns = x, data = result)
Prediction.to_csv("results.csv", index = False)
Prediction