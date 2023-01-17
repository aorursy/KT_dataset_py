# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import riiideducation

from sklearn.metrics import roc_auc_score



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from collections import Counter



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, learning_curve



import xgboost as xgb

env = riiideducation.make_env()





                 

import pandas as pd

train= pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',

               usecols=[1, 2, 3,7,8,9], dtype={'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16','answered_correctly':'int8','prior_question_elapsed_time': 'float32','prior_question_had_explanation': 'boolean'})



print(train.shape)
train = train.sort_values(['timestamp'], ascending=True)



train.drop(['timestamp'], axis=1,   inplace=True)



results_c = train.iloc[0:9000000,:][['content_id','answered_correctly']].groupby(['content_id']).agg(['mean'])

results_c.columns = ["answered_correctly_content"]



results_u = train.iloc[0:9000000,:][['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum'])

results_u.columns = ["answered_correctly_user", 'sum']
X = train.iloc[900000:990000,:]

X = pd.merge(X, results_u, on=['user_id'], how="left")

X = pd.merge(X, results_c, on=['content_id'], how="left")

X=X[X.answered_correctly!= -1 ]

X=X.sort_values(['user_id'])

Y = X[["answered_correctly"]]

X = X.drop(["answered_correctly"], axis=1)

X.head()
X.value_counts()
X.tail()
Y.head()
from sklearn.preprocessing import LabelEncoder



lb_make = LabelEncoder()

X['prior_question_had_explanation_enc'] = lb_make.fit_transform(X['prior_question_had_explanation'])
X = X[['answered_correctly_user', 'answered_correctly_content', 'sum','prior_question_elapsed_time','prior_question_had_explanation_enc']] 

X.fillna(0.5,  inplace=True)
from  sklearn.model_selection import train_test_split

Xt, Xv, Yt, Yv = train_test_split(X, Y, test_size =0.2, shuffle=False)



# Cross validate model with Kfold stratified cross val

kfold = StratifiedKFold(n_splits=5)
import numpy as np



Yt = np.ravel(Yt)
from sklearn.model_selection import RandomizedSearchCV
### META MODELLING  WITH ADABOOST, RF, EXTRATREES and GRADIENTBOOSTING



# Adaboost

DTC = DecisionTreeClassifier()



adaDTC = AdaBoostClassifier(DTC, random_state=7)



ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],

              "base_estimator__splitter" :   ["best", "random"],

              "algorithm" : ["SAMME","SAMME.R"],

              "n_estimators" :[1,2,5,10],

              "learning_rate":  [0.0001, 0.001,0.01]}



gsadaDTC = RandomizedSearchCV(adaDTC,param_distributions = ada_param_grid, cv=kfold,n_jobs = 4, scoring="accuracy", verbose = 1)



gsadaDTC.fit(Xt,Yt)



ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
#ExtraTrees 

ExtC = ExtraTreesClassifier()





## Search grid for optimal parameters

ex_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 5],

              "min_samples_split": [2, 3,5],

              "min_samples_leaf": [1, 3,5],

              "bootstrap": [False],

              "n_estimators" :[100,150,200],

              "criterion": ["gini"]}





gsExtC = RandomizedSearchCV(ExtC,param_distributions = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsExtC.fit(Xt,Yt)



ExtC_best = gsExtC.best_estimator_



# Best score

gsExtC.best_score_
# RFC Parameters tunning 

RFC = RandomForestClassifier()





## Search grid for optimal parameters

rf_param_grid = {"max_depth": [None],

              "max_features": [1, 3, 5],

              "min_samples_split": [2, 3, 5],

              "min_samples_leaf": [1, 3, 5],

              "bootstrap": [True],

              "n_estimators" :[100, 200, 300],

              "criterion": ["gini"]}





gsRFC = RandomizedSearchCV(RFC,param_distributions = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsRFC.fit(Xt,Yt)



RFC_best = gsRFC.best_estimator_



# Best score

gsRFC.best_score_
# Gradient boosting tuning



GBC = GradientBoostingClassifier()

gb_param_grid = {'loss' : ["deviance"],

              'n_estimators' : [100,200],

              'learning_rate': [0.1, 0.05, 0.01],

              'max_depth': [4, 8],

              'min_samples_leaf': [100,150],

              'max_features': [0.3, 0.1] 

              }



gsGBC = RandomizedSearchCV(GBC,param_distributions = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)



gsGBC.fit(Xt,Yt)



GBC_best = gsGBC.best_estimator_
# Best score

gsGBC.best_score_
votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)



votingC = votingC.fit(Xt, Yt)
y_pred = votingC.predict(Xv)

y_true = np.array(Yv)

roc_auc_score(y_true, y_pred)
test =  pd.read_csv('/kaggle/input/riiid-test-answer-prediction/example_test.csv')

test["prior_question_had_explanation_enc"] = lb_make.fit_transform(test["prior_question_had_explanation"])

test = pd.merge(test, results_u, on=['user_id'],  how="left")

test = pd.merge(test, results_c, on=['content_id'],  how="left")

test[['answered_correctly_user', 'answered_correctly_content', 'sum','prior_question_elapsed_time','prior_question_had_explanation_enc']]

test.fillna(0.5, inplace=True)



y_pred = votingC.predict(test[['answered_correctly_user', 'answered_correctly_content', 'sum','prior_question_elapsed_time','prior_question_had_explanation_enc']])



test['answered_correctly'] = y_pred



results_c = train[['content_id','answered_correctly']].groupby(['content_id']).agg(['mean'])

results_c.columns = ["answered_correctly_content"]



results_u = train[['user_id','answered_correctly']].groupby(['user_id']).agg(['mean', 'sum'])

results_u.columns = ["answered_correctly_user", 'sum']
iter_test = env.iter_test()

for (test_df, sample_prediction_df) in iter_test:

    test_df = pd.merge(test_df, results_u, on=['user_id'],  how="left")

    test_df = pd.merge(test_df, results_c, on=['content_id'],  how="left")

    test_df['answered_correctly_user'].fillna(0.5, inplace=True)

    test_df['answered_correctly_content'].fillna(0.5, inplace=True)

    test_df['sum'].fillna(0, inplace=True)

    test_df['prior_group_answers_correct'].fillna(0.5,inplace=True)

    test_df['prior_group_responses'].fillna(0.5,inplace=True)

    test_df['prior_question_elapsed_time'].fillna(0,inplace=True)   

    test_df['prior_question_had_explanation'].fillna(False, inplace=True)

    test_df["prior_question_had_explanation_enc"] = lb_make.fit_transform(test_df["prior_question_had_explanation"])

    test_df['answered_correctly'] =  votingC.predict(test_df[['answered_correctly_user', 'answered_correctly_content', 'sum','prior_question_elapsed_time','prior_question_had_explanation_enc']])

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])

test_df