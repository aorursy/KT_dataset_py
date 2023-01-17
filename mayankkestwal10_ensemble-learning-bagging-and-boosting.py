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
import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
df = pd.read_csv('../input/mnist_train.csv')
df.head()
df.describe()
df.shape
test = pd.read_csv('../input/mnist_test.csv')
test.head()
test.shape
xtr = df.iloc[:,1:]

ytr = df.iloc[:,0]



xtst = test.iloc[:,1:]

ytst = test.iloc[:,0]
#decision tree

classifier = DecisionTreeClassifier()

classifier.fit(xtr,ytr)
classifier.score(xtr,ytr)  #Overfitting
from sklearn.metrics import accuracy_score

pred = classifier.predict(xtst)

accuracy_score(pred,ytst)
#Ensemble



#Random Forest  -  Combination of various Decision tree

rf = RandomForestClassifier(n_estimators=10)

rf.fit(xtr,ytr)
rf.score(xtr,ytr)
rf.score(xtst,ytst)
# Bagging



bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5, max_features=1.0, n_estimators=20)

bg.fit(xtr,ytr)
bg.score(xtst,ytst)
bg.score(xtr,ytr)
#Boosting

ad = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=10, learning_rate=0.01)

ad.fit(xtr,ytr)
ad.score(xtst,ytst)
ad.score(xtr,ytr)
# Voting Classifier - Multiple model ensemble

lr = LogisticRegression()

tree = DecisionTreeClassifier()

svm = SVC(kernel='poly',degree=2)
evc = VotingClassifier( estimators= [('lr',lr),('tree',tree),('svm',svm)], voting = 'hard')
evc.fit(xtr.iloc[1:4000],ytr.iloc[1:4000])
evc.score(xtst,ytst)