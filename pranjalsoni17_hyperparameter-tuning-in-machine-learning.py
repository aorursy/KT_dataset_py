import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sklearn import datasets #import datasets to use in this notebook

from sklearn import model_selection #model_selection provides the methods for hyperparameter tuning

from skopt import BayesSearchCV #importing BayesSearchCV for Bayesian Optimization
import sklearn

sklearn.metrics.SCORERS.keys()
#loading bostonhousing dataset for regression problems

boston_data = datasets.load_boston()

r_X = pd.DataFrame(boston_data.data,columns=boston_data.feature_names)

r_y = boston_data.target

print(f"Shape of the dataset : {r_X.shape}")

print(r_X.head())



#applying standard scaling on the dataset

from sklearn.preprocessing import StandardScaler

sc_r = StandardScaler()

sc_r_X = sc_r.fit_transform(r_X)



#split the dataset into test and train 

from sklearn.model_selection import train_test_split

#taking 20% data as test data and 80% data as train data

r_X_train,r_X_test,r_Y_train,r_Y_test = train_test_split(sc_r_X,r_y, test_size=0.20, random_state=42) 
#loading Iris dataset for regression problems

iris_data = datasets.load_iris()

c_X = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)

c_y = iris_data.target

print(f"Shape of the dataset : {c_X.shape}")

print(c_X.head())



#applying standard scaling on the dataset

from sklearn.preprocessing import StandardScaler

sc_c = StandardScaler()

sc_c_X = sc_c.fit_transform(c_X)



#split the dataset into test and train 

from sklearn.model_selection import train_test_split

#taking 20% data as test data and 80% data as train data

c_X_train,c_X_test,c_Y_train,c_Y_test = train_test_split(sc_c_X,c_y, test_size=0.20, random_state=42) 
from sklearn.linear_model import Ridge

R_reg = Ridge()

#parameter grid dictionary

params = {

    'alpha' : [0.01, 0.1, 1.0, 10, 100],

    'fit_intercept' : [True,False],

    'normalize' : [True,False]

}

#r2_score is choosen as evaluation matrics

gs = model_selection.GridSearchCV(estimator=R_reg,param_grid=params,scoring='r2',cv=3,n_jobs = -1)

print(gs)

#fit our traing data to GridSearchCV

gs.fit(r_X_train,r_Y_train)

best_est = gs.best_estimator_

print(f"Best Estimator is : {best_est}")
from sklearn import svm

svm_clf = svm.SVC(probability=True)

#parameter grid dictionary

params = {

    'C': [0.1, 1, 10, 100, 1000],  

    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 

    'kernel': ['linear','rbf'] #other kernels are ‘sigmoid’, ‘precomputed’, 'ploy'

    

}

#neg_log_loss is choosen as evaluation matrics

#here I am using n_iter = 20 only you can choose more

gs = model_selection.RandomizedSearchCV(n_iter=20,estimator=svm_clf,param_distributions=params,scoring='neg_log_loss',cv=3,n_jobs = -1)

print(gs)

#fit our traing data to RandomizedSearchCV

gs.fit(c_X_train,c_Y_train)

best_est = gs.best_estimator_

print(f"Best Estimator is : {best_est}")
#importing skopt library for optimization

import skopt

from sklearn import ensemble



rf_clf = ensemble.RandomForestClassifier()

#parameter grid dictionary

params = {

   'n_estimators': [int(x) for x in np.linspace(50,1000, num=20)], #this return a list with 20 equally spaced numbers between 50 and 1000

   'max_features': [len(c_X.columns), "sqrt", "log2"],

   'max_depth': [int(x) for x in np.linspace(5,50,num=10)],

   'min_samples_split': [3,4,6,7,8,9,10],

   'min_samples_leaf': [1,3,5,7,9,10],

   'bootstrap': [True,False]

    

}



#here I am using n_iter = 50 only you can choose more

gs = skopt.BayesSearchCV(n_iter=50,estimator=rf_clf,search_spaces=params,optimizer_kwargs={'base_estimator': 'RF'},cv=5,n_jobs = -1)

print(gs)

#fit our traing data to BayesSearchCV

gs.fit(c_X_train,c_Y_train)

best_est = gs.best_estimator_

best_score = gs.best_score_

print(f"Best Estimator is : {best_est}")

print(f"Best Score is : {best_score}")