# Loading the packages

import numpy as np

import pandas as pd 

from sklearn.dummy import DummyClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
# Loading the training dataset

df_train = pd.read_csv("../input/train.csv")
y = df_train["target"]

# We exclude the target and id columns from the training dataset

df_train.pop("target");

df_train.pop("id")

colnames = df_train.columns

X = df_train 

del df_train

X = X.values # Converting pandas dataframe to numpy array 

y = y.values # Converting pandas series to numpy array 
from sklearn.metrics import make_scorer 

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



# We are going to perform a gridsearch 



# TODO: Initialize the classifier

clf = RandomForestClassifier(class_weight = 'balanced', random_state=0)



# Create the parameters list you wish to tune, using a dictionary if needed.

# parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}



parameters = {'n_estimators':[ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 

              'max_depth':[ 1, 2, 3, 4, 5, 6 ],

              'criterion':['gini', 'entropy'], 

              'max_features': ['auto', 'sqrt', 'log2'], 

              

             }



# Make an roc_auc scoring object using make_scorer()

scorer = make_scorer(roc_auc_score)



grid_obj = GridSearchCV(clf, parameters, scoring=scorer, verbose=1, cv=10)



grid_fit = grid_obj.fit(X, y)

# Get the estimator

best_clf = grid_fit.best_estimator_

print(best_clf)
model = RandomForestClassifier(bootstrap=True, class_weight='balanced',

            criterion='entropy', max_depth=2, max_features='log2',

            max_leaf_nodes=None, min_impurity_decrease=0.0,

            min_impurity_split=None, min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            n_estimators=6, n_jobs=None, oob_score=False, random_state=0,

            verbose=0, warm_start=False)
model.fit(X, y)
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(model, random_state=0).fit(X, y)



eli5.show_weights(perm, feature_names = colnames.tolist())
df_test = pd.read_csv("../input/test.csv")

df_test.pop("id");

X = df_test 

del df_test

y_pred = model.predict_proba(X)

y_pred = y_pred[:,1]

# submit prediction

smpsb_df = pd.read_csv("../input/sample_submission.csv")

smpsb_df["target"] = y_pred

smpsb_df.to_csv("random_forests.csv", index=None)