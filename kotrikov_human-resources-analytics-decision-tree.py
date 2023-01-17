# Let's import few basics

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv('../input/HR_comma_sep.csv')
data.info()
data.head().T
# Lets create dummies for sales and salary variables

data=pd.concat([data,pd.get_dummies(data['sales'],prefix='sales'),pd.get_dummies(data['salary'],prefix='salary')],axis=1)
# set of independent variables

X=data.drop(['sales','salary','left'],axis=1)
# dependent variable

y=data['left']
from sklearn.model_selection import train_test_split
# train test split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)
X_test.shape,X_train.shape
# In my opinion it is important to check for train/test datasets equality 

y_train.mean(),y_test.mean()
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=17)
# fitting model

tree.fit(X_train,y_train)
from sklearn.metrics  import accuracy_score
accuracy_score(y_test,tree.predict(X_test))
from sklearn.model_selection import GridSearchCV
# finding best parameters

tree_params={'max_depth': range(1, 21)}
tree=DecisionTreeClassifier()
tree_search=GridSearchCV(tree,tree_params,cv=5,n_jobs=-1,verbose=1)
tree_search.fit(X_train,y_train)
tree_search.best_params_
tree_search.best_score_
tree_search.cv_results_['mean_test_score']
# so, the best number for max depth is 8

plt.scatter(tree_params['max_depth'],tree_search.cv_results_['mean_test_score']);