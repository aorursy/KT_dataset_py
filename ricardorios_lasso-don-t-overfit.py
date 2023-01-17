# Loading the packages

import numpy as np

import pandas as pd 

from sklearn.preprocessing import StandardScaler

from scipy import stats

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import make_scorer 

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')
# Loading the training dataset

df_train = pd.read_csv("../input/train.csv")

y = df_train["target"]

# We exclude the target and id columns from the training dataset

df_train.pop("target");

df_train.pop("id")

colnames1 = df_train.columns

scaler = StandardScaler()

scaler.fit(df_train)

X = scaler.transform(df_train)

df_train = pd.DataFrame(data = X, columns=colnames1)
# Find best hyperparameters (roc_auc)

random_state = 0

clf = LogisticRegression(random_state = random_state)

param_grid = {'class_weight' : ['balanced'], 

              'penalty' : ['l1'],  

              'C' : [0.0001, 0.0005, 0.001, 

                     0.005, 0.01, 0.05, 0.1, 0.5, 1, 

                     10, 100, 1000, 1500, 2000, 2500, 

                     2600, 2700, 2800, 2900, 3000, 3100, 3200  

                     ] , # This hyperparameter is lambda 

              'max_iter' : [100, 1000, 2000, 5000, 10000] }



# Make an roc_auc scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



grid = GridSearchCV(estimator = clf, param_grid = param_grid , 

                    scoring = scorer, verbose = 1, cv=20,

                    n_jobs = -1)





X = df_train.values



grid.fit(X,y)



print("Best Score:" + str(grid.best_score_))



best_parameters = grid.best_params_

# We are going to print the hyperparameters of the best model 

best_clf = grid.best_estimator_

print(best_clf)
model = LogisticRegression(C=0.1, class_weight='balanced', dual=False,

          fit_intercept=True, intercept_scaling=1, max_iter=100,

          multi_class='warn', n_jobs=None, penalty='l1', random_state=0,

          solver='warn', tol=0.0001, verbose=0, warm_start=False);



model.fit(X, y);

df_test = pd.read_csv("../input/test.csv")

df_test.pop("id");

X = df_test 

X = scaler.transform(X)

df_test = pd.DataFrame(data = X, columns=colnames1)  

X = df_test.values



y_pred = model.predict_proba(X)

y_pred = y_pred[:,1]  
# submit prediction

smpsb_df = pd.read_csv("../input/sample_submission.csv")

smpsb_df["target"] = y_pred

smpsb_df.to_csv("submission.csv", index=None)
