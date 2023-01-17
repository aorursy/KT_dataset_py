import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Lasso





from IPython.display import SVG

from IPython.display import display



from IPython.display import HTML

from sklearn import  tree







# HTTP Requests 

import requests



# Read and write strings as files

from io import StringIO







# Date Time

import datetime

test = pd.read_csv('test.csv',header=None)

test = pd.DataFrame(test)



train = pd.read_csv('train.csv',header=None)

train = pd.DataFrame(train)



train_labels = pd.read_csv('train_labels.csv',header=None)

train_labels = pd.DataFrame(train_labels)
test =  test.fillna(test.median())

train = train.fillna(train.median())
#X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)

clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

#clf=Lasso(alpha=0, fit_intercept=True, normalize=False, precompute=False,selection='cyclic')

#clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

#clf = LogisticRegression(random_state=0, solver='newton-cg',multi_class='multinomial')

clf.fit(train,train_labels)



y_predict =clf.predict(test)

y_predict = pd.DataFrame(y_predict)

y_predict.to_csv('out.csv', index=False)