import numpy as np

import pandas as pd

import seaborn as sns



import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
feats = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
all_data = pd.concat((train.loc[:,feats],

                      test.loc[:,feats]))

all_data['Sex'] = all_data['Sex'].map({'male':0, 'female':1})

all_data = pd.get_dummies(all_data)
X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y_train = train.Survived
from sklearn.model_selection import GridSearchCV

xgb_model = xgb.XGBClassifier()

clf = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6],

                    'learning_rate':[0.1],

                    'n_estimators': [50,100,200]}, verbose=1)

clf.fit(X_train, y_train)

print(clf.best_score_)

print(clf.best_params_)
xgb_model = xgb.XGBClassifier()

clf = GridSearchCV(xgb_model,

                   {'max_depth': [2,4,6,8],

                    'learning_rate':[0.1],

                    'n_estimators': [50,100,200,300]}, verbose=1).fit(X_train, y_train)

print(clf.best_score_)

print(clf.best_params_)
print()
result = clf.cv_results_

score_list =  ['param_learning_rate', 'param_max_depth', 'param_n_estimators', 'params', 'mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time', 'mean_test_score', 'std_test_score', 'mean_train_score', 'std_train_score', 'rank_test_score']

result = {k:v for (k,v) in result.items() if k in score_list}

df = pd.DataFrame(result, columns=score_list)

df
import xgboost as xgb



dtrain = xgb.DMatrix(X_train, label = y_train)

dtest = xgb.DMatrix(X_test)



params = {'objective': 'binary:logistic',

          'eval_metric': 'error',

          'eta': 0.01,

          'max_depth': 9,

          'silent': 1

          }



cv_result = xgb.cv(params, dtrain, num_boost_round=5000, nfold=5,

                   callbacks=[xgb.callback.print_evaluation(show_stdv=True),

                              xgb.callback.early_stop(100)])