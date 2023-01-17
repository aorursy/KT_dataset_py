import pandas as pd

import sklearn as sk

from sklearn.model_selection import GridSearchCV



from sklearn.model_selection import RandomizedSearchCV

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

import numpy as np

from sklearn.linear_model import SGDClassifier


train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        na_values="nulo")



test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        na_values="nulo")





x_train1 = train_data[["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country"] ]

x_teste = test_data[["age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country"] ]



y_train = train_data[["income"]]





frames = [x_train1, x_teste]

result = pd.concat(frames, sort = False)

result = pd.get_dummies(result)

x_train = result.head(32560)



frames = [x_teste, x_train1]

result = pd.concat(frames, sort = False)

result = pd.get_dummies(result)

x_test = result.head(16280)
SGD = SGDClassifier()

param_grid = {"loss" : ["hinge", "modified_huber", "log"],

              "penalty" :   ["l2", "l1", "elasticnet"],

             }



clf = GridSearchCV(SGD,param_grid=param_grid , cv =6, n_jobs = -1, verbose=2, refit = False)

clf.fit(x_train, y_train)

clf.best_score_
'''

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 150)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 40)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 20, cv = 6, verbose=2, random_state=42, n_jobs = -1)



rf_random.fit(x_train, y_train)

rf_random.best_score_

'''
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier



param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "n_estimators": [1, 2]

             }

DTC = DecisionTreeClassifier(random_state = 11, max_features = "auto",max_depth = None)

ABC = AdaBoostClassifier(base_estimator = DTC)



# run grid search

grid_search_ABC = GridSearchCV(ABC, param_grid=param_grid, scoring = 'roc_auc', cv = 6, n_jobs =  -1, verbose=2)

grid_search_ABC.fit(x_train, y_train)

grid_search_ABC.best_score_
#rf_random.best_params_



# treinando com todos os dados:



classificador = RandomForestClassifier(n_estimators= 1601,

 min_samples_split= 10,

 min_samples_leaf= 2,

 max_features= 'auto',

 max_depth= 63,

 bootstrap= False)



classificador = classificador.fit(x_train, y_train)
out = classificador.predict(x_test)

df = pd.DataFrame(out, columns=['Income'])

df.to_csv("submission.csv", index_label = 'Id')