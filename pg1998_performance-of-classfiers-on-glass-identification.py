# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


data=pd.read_csv('/kaggle/input/glass/glass.csv')
data.head()
data.info()
data.columns
data['Type'].value_counts()
X=data.iloc[:,0:9]

y=data.iloc[:,9]

print(X.shape)

print(y.shape)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
dtc=DecisionTreeClassifier()

rfc=RandomForestClassifier(n_estimators=100)

lr=LogisticRegression()

lr_mul=LogisticRegression(solver='newton-cg',multi_class='multinomial')
dtc_accuracy=cross_val_score(dtc,X,y,scoring='accuracy',cv=8).mean()

print('accuracy of decision tree classifier is',dtc_accuracy)
rfc_accuracy=cross_val_score(rfc,X,y,scoring='accuracy',cv=8).mean()

print('accuracy of random forest classifier is',rfc_accuracy)
lr_accuracy=cross_val_score(lr,X,y,scoring='accuracy',cv=8).mean()

print('accuracy of logistic regression is',lr_accuracy)
lr1_accuracy=cross_val_score(lr_mul,X,y,scoring='accuracy',cv=8).mean()

print('accuracy of decision tree classifier is',lr1_accuracy)
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
classifier_pipeline = make_pipeline(MinMaxScaler(),LogisticRegression(solver='newton-cg',multi_class='multinomial'))

lr2_accuracy=cross_val_score(classifier_pipeline,X,y,scoring='accuracy',cv=8).mean()

print('accuracy of logistic  Regression is',lr2_accuracy)
from sklearn.naive_bayes import GaussianNB

gnb_accuracy=cross_val_score(GaussianNB(),X,y,scoring='accuracy',cv=8)

print(type(gnb_accuracy))

print('Accuracy of naive bayes classifier is',gnb_accuracy.mean())
import random
def repeat_cv(X,y,model):

    n_rep=10

    accuracy_scores=np.zeros(n_rep)

    for i in range(n_rep):

        model_accuracy=cross_val_score(model,X,y,scoring='accuracy',cv=8)

        accuracy_scores[i] = model_accuracy.mean()

    return accuracy_scores    

        
rf_acc=repeat_cv(X,y,rfc)

print(rf_acc.mean())
dtc_acc=repeat_cv(X,y,dtc)

print(dtc_acc.mean())
lr_acc=repeat_cv(X,y,lr_mul)

print(lr_acc.mean())
gnb_acc=repeat_cv(X,y,GaussianNB())

print(gnb_acc.mean())