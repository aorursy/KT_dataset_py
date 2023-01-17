import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()
X=cancer.data

y=cancer.target
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)
#using SVC with default paramters 
from sklearn.svm import SVC

svc= SVC()

svc.fit(X_train,y_train)
svc.score(X_test,y_test)
#GridsearchCV

from sklearn.model_selection import GridSearchCV

tune_param=({"C":[1.0,2.0],"kernel":["linear","rbf"],"degree":[0,1,2]})

gsv=GridSearchCV(estimator=svc,param_grid=tune_param)

                                 
gsv.fit(X_train,y_train)
#best parameters and scores after tuning parameters.
gsv.best_score_
gsv.best_params_
#lets use these parameters and again check the model for score.
svc= SVC(C=2,degree=0,kernel="linear")

svc.fit(X_train,y_train)
svc.score(X_test,y_test)
# So the model accuracy increased by doing the parameter tuning.
#RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

tune_param=({"C":[1.0,2.0],"kernel":["linear","rbf"],"degree":[0,1,2],"gamma":[0.1,0.2,0.3,0.4,0.5]})

rscv=RandomizedSearchCV(estimator=svc,param_distributions=tune_param,n_iter=10,cv=10)#cv MEANS cross validation
rscv.fit(X_train,y_train)
rscv.best_params_
rscv.best_score_
#lets use the bestparameters and see how model accuracy increases.

svm=SVC(C=1.0,kernel="linear",gamma=0.1,degree=0)

svm.fit(X_train,y_train)
svm.score(X_test,y_test)