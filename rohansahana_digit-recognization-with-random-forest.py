import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split        
from sklearn.preprocessing import StandardScaler           
from sklearn.linear_model import LogisticRegression       

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/digit-recognizer/train.csv')
data
X = data.iloc[:,1:].values        
Y = data.iloc[:,0].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
# X_train = X
# Y_train = Y
# # Fitting Random Forest Classification to the Training set
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import RandomizedSearchCV

# est = RandomForestClassifier(n_jobs=-1)
# rf_p_dist={'max_depth':[3,5,10,None],
#               'n_estimators':[100, 500, 1000, 3000],
#                'criterion':['gini','entropy'],
#                'bootstrap':[True,False],
#               }

# def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
#     rdmsearch = RandomizedSearchCV(est, param_distributions = p_distr, n_jobs = -1, cv = 9, )
#     #CV = Cross-Validation ( here using Stratified KFold CV)
#     rdmsearch.fit(X,Y)
#     ht_params = rdmsearch.best_params_
#     ht_score = rdmsearch.best_score_
#     return ht_params, ht_score

# rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dist, 3000, X, Y)

from sklearn.model_selection import GridSearchCV
est = RandomForestClassifier()
parameters = [{'criterion':['gini'], 'n_estimators':[3000]}, 
              {'criterion':['entropy'], 'n_estimators':[3000]}]
grid_search = GridSearchCV(estimator = est, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, Y_train)
# # Fitting Random Forest Classification to the Training set
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 50)
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix,accuracy_score
# cm = confusion_matrix(y_test, y_pred)

# accuracy_score=accuracy_score(y_test,y_pred)

# #claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=300,bootstrap= True,criterion='entropy',max_depth=3,max_features=2,min_samples_leaf= 3)

# ## Cross Validation good for selecting models
# from sklearn.model_selection import cross_val_score

# cross_val=cross_val_score(claasifier,X,y,cv=10,scoring='accuracy').mean()