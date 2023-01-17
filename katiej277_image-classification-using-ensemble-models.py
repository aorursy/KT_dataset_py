import pandas as pd

import numpy as np

import os



from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.model_selection import train_test_split



os.listdir('../input/')

train = pd.read_csv('../input/fashion-mnist_train.csv')

test = pd.read_csv('../input/fashion-mnist_test.csv')



ds = train.append(test)



X = ds.iloc[:,1:]

y = ds.iloc[:,0]



X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = True)
rf = RandomForestClassifier(n_estimators = 100,max_features = 5)



rf.fit(X_train,y_train)



print('Training score: {:.3f}'.format(rf.score(X_train,y_train)))

print('Test score : {:.3f}'.format(rf.score(X_test,y_test)))
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score



rf = RandomForestClassifier().fit(X_train,y_train)





param_values = {'n_estimators' : [10,50,100], 'max_features' : [5,10,20]}

gs = GridSearchCV(rf,param_grid = param_values, cv = 5,scoring = 'accuracy')



gs.fit(X_test,y_test)

print(gs.best_score_,gs.best_params_)
gb = GradientBoostingClassifier(n_estimators = 100,max_features = 5,learning_rate = 0.1)



gb.fit(X_train,y_train)

print('Training score: {:.3f}'.format(gb.score(X_train,y_train)))

print('Test score: {:.3f}'.format(gb.score(X_test,y_test)))