# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/creditcard.csv")

df.head()
df.info()
df.isnull().sum() # checked for missing value, no value is missing
#df[(df.Class==0)].count() #284315- genuine

df[(df.Class==1)].count() #492 -fraud transactions
#lets divide the data into features ans labels and drop time as its not a good feature to determine the transaction 

#as fraud or genuine
# lets take the data and divide in in test and train

X=df.drop(columns=["Time","Class"],axis="columns")

y=df.Class

#from sklearn.preprocessing import Normalizer

#Normalizer(X,y)
X
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
X_train.shape
#variance Threshold for feature selection

from sklearn.feature_selection import VarianceThreshold

var = VarianceThreshold(threshold=.5)

var.fit(X_train,y_train)

X_train_var=var.transform(X_train)

X_test_var=var.transform(X_test)
X_train_var.shape
mask=var.get_support()

mask

plt.matshow(mask.reshape(-1,1))
from sklearn.feature_selection import SelectPercentile

select = SelectPercentile(percentile=50)

select.fit(X_train,y_train)

X_train_select=select.transform(X_train)

X_test_select=select.transform(X_test)
X_train_select.shape  # features selected are 50 %  from 28 to 14
mask=select.get_support()

mask

plt.matshow(mask.reshape(1,-1))
mask  # the yellow one are selected featues
from sklearn.feature_selection import SelectKBest

skbest = SelectKBest(k=10)

skbest.fit(X_train,y_train)

X_train_skbest=skbest.transform(X_train)

X_test_skbest=skbest.transform(X_test)
mask=skbest.get_support()

mask

plt.matshow(mask.reshape(1,-1))

#yellow one are the feature used.
X_train_skbest.shape
#choose a classification model # logistic regression

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

logreg.score(X_test,y_test)
logreg=LogisticRegression()

logreg.fit(X_train_select,y_train)
logreg.score(X_test_select,y_test)
from sklearn.linear_model import RidgeClassifier

logreg=RidgeClassifier()

logreg.fit(X_train_select,y_train)

logreg.score(X_test_select,y_test)
from sklearn.linear_model import RidgeClassifier

logreg=RidgeClassifier()

logreg.fit(X_train_skbest,y_train)

logreg.score(X_test_skbest,y_test)
from sklearn.linear_model import RidgeClassifier

logreg=RidgeClassifier()

logreg.fit(X_train_var,y_train)

logreg.score(X_test_var,y_test)
from sklearn.ensemble import GradientBoostingClassifier

logreg=GradientBoostingClassifier()

logreg.fit(X_train_select,y_train)

logreg.score(X_test_select,y_test)