import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression
data = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
data.head()
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1)



y = data['diagnosis']
# X.isnull().sum()
X.shape, y.shape
# lr = LogisticRegression()

lr = LogisticRegression(max_iter=5000)
lr.fit(X,y)
lr.score(X,y)
params = [    

    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],   # Used to specify the norm used in the penalization.

    'C' : np.logspace(-4, 4, 20),                      # Inverse of regularization strength; must be a positive float.

    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],  # Algorithm to use in the optimization problem.

    'max_iter' : [100, 1000,2500, 5000]                # Maximum number of iterations taken for the solvers to converge.

    }

]



# There are many other parameters that we could use... but for nw will start with this.
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(estimator = lr, param_grid = params, scoring = 'accuracy', cv = 3, verbose=True, n_jobs=-1)

# cv --> Determines the cross-validation splitting strategy

# verbose --> Controls the verbosity. Verbose is a general programming term for produce lots of logging output. You can think of it as asking the program to "tell me everything about what you are doing all the time". 

# n_jobs --> Number of jobs to run in parallel. `-1` means using all processors. 
clf_fit = clf.fit(X,y)
# Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data.

clf_fit.best_estimator_
clf_fit.score(X,y)

# Returns the score on the given data.

# This uses the score defined by scoring where provided, and the best_estimator_.score method otherwise.
# Mean cross-validated score of the best_estimator

clf_fit.best_score_