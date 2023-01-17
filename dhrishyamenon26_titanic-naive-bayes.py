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
import pandas as pd
import numpy as np
titan = pd.read_csv("../input/Titanic.csv")
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
titan.head(10)
ip_columns = ["PassengerId","Survived","Pclass","Age","SibSp","Parch","Fare"]
op_column  = ["Sex"]
# Splitting data into train and test

Xtrain,Xtest,ytrain,ytest = train_test_split(titan[ip_columns],titan[op_column],test_size=0.3, random_state=0)
ignb = GaussianNB()
imnb = MultinomialNB()
# Building and predicting at the same time 



pred_gnb = ignb.fit(Xtrain,ytrain).predict(Xtest)
pred_mnb = imnb.fit(Xtrain,ytrain).predict(Xtest)
confusion_matrix(ytest,pred_gnb)
pd.crosstab(ytest.values.flatten(),pred_gnb)
np.mean(pred_gnb==ytest.values.flatten()) 
confusion_matrix(ytest,pred_mnb)
pd.crosstab(ytest.values.flatten(),pred_mnb)
np.mean(pred_mnb==ytest.values.flatten())
confusion_matrix(ytest,pred_mnb)