import pandas as pd

import numpy as np

import sklearn
import warnings

warnings.filterwarnings("ignore")

import time

from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,VotingClassifier)

import lightgbm as lgb

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer
train = pd.read_csv('train.csv')

test = pd.read_csv('test.csv')
train = train[train.chem_2<=10]

train = train[train.chem_7<=12]

train = train[(train.attribute <= 2.4) & (train.attribute>=1.4)]
train.corr()['class']
train.drop(['chem_3','chem_5','attribute'],axis=1,inplace=True)

test.drop(['chem_3','chem_5','attribute'],axis=1,inplace=True)
temp = train.copy()

y = train['class'].copy()

temp.drop(['id','class'],axis=1,inplace=True)
X_train,X_val,y_train,y_val = train_test_split(temp,y,test_size=0.10,shuffle=False)
estimators = [('rf', RandomForestClassifier(max_depth= 10, n_estimators= 300)), 

              ('gb', GradientBoostingClassifier(max_depth= 5, n_estimators= 300)),

              ('xgb', XGBClassifier(learning_rate= 0.5, max_depth= 10, n_estimators= 300)),

              ('lgb', LGBMClassifier(learning_rate= 0.05, max_depth= 5, n_estimators= 300))]



soft_voter = VotingClassifier(estimators=estimators, voting='soft')

hard_voter = VotingClassifier(estimators=estimators, voting='hard')
soft_voter.fit(X_train,y_train)

hard_voter.fit(X_train,y_train)



soft_acc = accuracy_score(y_val,soft_voter.predict(X_val))

hard_acc = accuracy_score(y_val,hard_voter.predict(X_val))



print("Acc of soft voting classifier:{}".format(soft_acc))

print("Acc of hard voting classifier:{}".format(hard_acc))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



parameters = {'weights':[[1,1,1,1],[1,1,1,2],[1,1,2,1],[1,2,1,1],[2,1,1,1],[1,2,2,2],[2,1,2,2],[2,2,1,2],[2,2,2,1]]}    #Dictionary of parameters

scorer = make_scorer(accuracy_score)         #Initialize the scorer using make_scorer

grid_obj = GridSearchCV(VotingClassifier(estimators=estimators, voting='soft'),parameters,scoring=scorer)         #Initialize a GridSearchCV object with above parameters,scorer and classifier

grid_fit = grid_obj.fit(X_train,y_train)        #Fit the gridsearch object with X_train,y_train

best_clf_sv = grid_fit.best_estimator_         #Get the best estimator. For this, check documentation of GridSearchCV object

unoptimized_predictions = (VotingClassifier(estimators=estimators, voting='soft').fit(X_train, y_train)).predict(X_val)      #Using the unoptimized classifiers, generate predictions

optimized_predictions = best_clf_sv.predict(X_val)        #Same, but use the best estimator



acc_unop = accuracy_score(y_val, unoptimized_predictions)*100       #Calculate accuracy for unoptimized model

acc_op = accuracy_score(y_val, optimized_predictions)*100         #Calculate accuracy for optimized model



print("Accuracy score on unoptimized model:{}".format(acc_unop))

print("Accuracy score on optimized model:{}".format(acc_op))
X_test = test.copy()

X_test.drop('id',axis=1,inplace=True)

final_preds = best_clf_sv.fit(temp,y).predict(X_test) 
sub = pd.DataFrame({'id':test.id,'class':final_preds})
sub.to_csv('sub7.csv',index=False)