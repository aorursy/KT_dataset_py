import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import Pipeline

from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from bayes_opt import BayesianOptimization
import eli5

from eli5.sklearn import PermutationImportance
import time

import os

import gc

import random

from scipy.stats import uniform
os.chdir("../input/")
data = pd.read_csv("winequalityN.csv")
data.head()

data.info()

data.shape
data.describe
data.isnull().values.any()

data.isnull().sum()

data.dropna(axis=0,inplace=True)
data.dropna(axis=0,inplace=True)
data.shape
sns.pairplot(data, diag_kind='scatter',hue='type')

X = data.iloc[ :, 1:14]
y=data.iloc[:,0]
y = y.map({'white':1, 'red' : 0})

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.35,

                                                    shuffle = True

                                                    )
pipe_list=[('scale',ss()),

           ('pca',PCA()),

		   ('xgb',XGBClassifier(n_jobs=2,

							    silent=False))

							 ]
pipe=Pipeline(pipe_list)
grid_parameters={'xgb__learning_rate':  [0.05, 0.07],

              'xgb__n_estimators':   [50,  100],

              'xgb__max_depth':      [3,5],

              'pca__n_components' : [5,7]

              }
grid_search=GridSearchCV(pipe,

						  grid_parameters,

						   cv=3,

						   n_jobs=3,

						   verbose=1,

						   scoring=['accuracy','roc_auc'],

						   refit='roc_auc')
start=time.time()

grid_search.fit(X_train,y_train)

stop=time.time()

(stop-start)/60
f"Best score: {grid_search.best_score_} "
f"Best parameter set {grid_search.best_params_}"
plt.bar(grid_search.best_params_.keys(), grid_search.best_params_.values(), color='r')

plt.xticks(rotation=45)
y_pred=grid_search.predict(X_test)

y_pred
accuracy=accuracy_score(y_test,y_pred)
f"Accuracy is{accuracy*100}%"
parameter_random = {'xgb__learning_rate':  uniform(0, 1),

                    'xgb__n_estimators':   range(50,100),

                    'xgb__max_depth':      range(3,5),

                    'pca__n_components' : range(5,7)}
random_search=RandomizedSearchCV(pipe,

                             param_distributions=parameter_random,

                             cv=3,

                             n_iter=27,

                             n_jobs=3,

                             verbose=1,

                             scoring=['accuracy','roc_auc'],

                             refit='roc_auc')
start=time.time()

random_search.fit(X_train,y_train)

stop=time.time()

(stop-start)/60
f"Best score: {random_search.best_score_} "
f"Best parameter set {random_search.best_params_}"
y_pred=random_search.predict(X_test)

y_pred
accuracy=accuracy_score(y_test,y_pred)
f"Accuracy is{accuracy*100}%"
plt.bar(random_search.best_params_.keys(), random_search.best_params_.values(), color='g')

plt.xticks(rotation=45)
parameter_bo={

           'learning_rate':  (0, 1),            

           'n_estimators':   (50,100),         

           'max_depth':      (3,5),            

           'n_components' :  (5,7)

            }