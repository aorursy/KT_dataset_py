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

from sklearn.preprocessing import PolynomialFeatures



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

df_train = pd.DataFrame(data = X, columns=colnames1)   # df_train is standardized 
random_forest_predictors = ["33", "279", "272", 

                           "83", "237", "241", 

                           "91", "199", "216", 

                           "19", "65", "141", "70", "243", "137", "26", "90"]



selected_predictors = [0, 2, 4, 6, 7, 10, 16]

new_predictors = []



for i in selected_predictors: 

    new_predictors.append(random_forest_predictors[i])



df_train = df_train[new_predictors]

poly = PolynomialFeatures(2, interaction_only=True)

poly.fit(df_train)

X = poly.transform(df_train)
# We adapt code from this kernel: 

# https://www.kaggle.com/vincentlugat/logistic-regression-rfe



# Find best hyperparameters (roc_auc)

random_state = 0

clf = LogisticRegression(random_state = random_state)

param_grid = {'class_weight' : ['balanced'], 

              'penalty' : ['l1'],  

              'C' : [0.0001, 0.0005, 0.001, 

                     0.005, 0.01, 0.05, 0.1, 0.5, 1, 

                     10, 100, 1000, 1500, 2000 

                     ], 

              'max_iter' : [100, 1000] }



# Make an roc_auc scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



grid = GridSearchCV(estimator = clf, param_grid = param_grid , 

                    scoring = scorer, verbose = 10, cv=20,

                    n_jobs = -1)







grid.fit(X,y)



print("Best Score:" + str(grid.best_score_))

print("Best Parameters: " + str(grid.best_params_))



best_parameters = grid.best_params_
# We get the best model 

best_clf = grid.best_estimator_

print(best_clf)
model = LogisticRegression(C=0.1, class_weight='balanced', dual=False,

          fit_intercept=True, intercept_scaling=1, max_iter=100,

          multi_class='warn', n_jobs=None, penalty='l1', random_state=0,

          solver='warn', tol=0.0001, verbose=0, warm_start=False);



model.fit(X, y);
print(model.coef_)
df_test = pd.read_csv("../input/test.csv")

df_test.pop("id");

X = df_test 

X = scaler.transform(X)

df_test = pd.DataFrame(data = X, columns=colnames1)   # df_train is standardized 

df_test = df_test[new_predictors]



X = poly.transform(df_test)

y_pred = model.predict_proba(X)

y_pred = y_pred[:,1]    
# submit prediction

smpsb_df = pd.read_csv("../input/sample_submission.csv")

smpsb_df["target"] = y_pred

smpsb_df.to_csv("logistic_regression_l2_v2.csv", index=None)