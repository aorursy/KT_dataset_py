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

%matplotlib inline
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
train_costa_rica = pd.read_csv("../input/train.csv")
test_costa_rica = pd.read_csv("../input/test.csv")
ntrain = train_costa_rica.dropna()
ntest = test_costa_rica.dropna()
ntrain.v2a1.value_counts().plot(kind='barh')
ntrain.hogar_nin.value_counts().plot(kind='barh')
#precisamos de dados em que haja vários representantes por classe
ntrain.describe()
vntrain = ntrain.apply(preprocessing.LabelEncoder().fit_transform) 
Xntrain = vntrain[["rooms","SQBage", "SQBovercrowding", "epared3", "instlevel9", 'idhogar', 'dependency', 'edjefe']]
Yntrain = vntrain.Target
#tentativa de descobrir o melhor k de 0 até 100
better=0
melhork=0
for c in range(1,101):
    knn = KNeighborsClassifier(n_neighbors=c)
    scores = cross_val_score(knn, Xntrain, Yntrain, cv=5)
    if(scores.mean() > better):
        melhork = c
        better = scores.mean()   
melhork
better
knn = KNeighborsClassifier(n_neighbors = melhork)
knn.fit(Xntrain,Yntrain)
vntest = ntest.apply(preprocessing.LabelEncoder().fit_transform)
Xntest = vntest[["rooms","SQBage", "SQBovercrowding", "epared3", "instlevel9",'idhogar', 'dependency', 'edjefe']]
YntestPred = knn.predict(Xntest)
YntestPred
prediction = pd.DataFrame(Xntest)
prediction['income'] = YntestPred
prediction.to_csv()
prediction.head()
prediction.to_csv("predictionpoverty.csv")