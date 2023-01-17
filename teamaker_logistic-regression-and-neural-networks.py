import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn import linear_model #for logistic regression

from sklearn.neural_network import MLPClassifier #for neural network

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict, validation_curve 

#GridSearchCV is used to optimize parameters of the models used

#the other modules and functions 

from sklearn.ensemble import VotingClassifier #for creating ensembles of classifiers



df = pd.read_csv('../input/data.csv', skiprows=[0], header=None)

df = df.replace({'B':0, 'M':1})

x = df.iloc[:,2:] 

y = df.iloc[:,1]

print (x.shape, y.shape)
x_mean = x.mean()

x_std = x.std()

x_norm = (x - x_mean)/x_std

print (x_norm.shape)
logreg = linear_model.LogisticRegression()

kfold = KFold(n_splits=5,random_state=7)

cv_results = cross_val_score(logreg, x_norm, y, cv=kfold)

print (cv_results.mean()*100, "%")
logreg = linear_model.LogisticRegression()

param_grid = {"C":[0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

grid = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=kfold)

grid.fit(x_norm,y)

print (grid.best_estimator_)

print (grid.best_score_*100, "%")
#plot validation curve

num_splits = 5

num_C_values = 10 # we iterate over 10 possible C values

logreg = linear_model.LogisticRegression()

kfold = KFold(n_splits=5,random_state=7)

C_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

train_scores, valid_scores = validation_curve(logreg, x_norm, y, "C", C_values, cv=kfold)

train_scores = pd.DataFrame(data=train_scores, index=np.arange(0, num_C_values), columns=np.arange(0,num_splits)) 

valid_scores = pd.DataFrame(data=valid_scores, index=np.arange(0, num_C_values), columns=np.arange(0,num_splits)) 

plt.semilogx(C_values, train_scores.mean(axis=1), label='training score')

plt.semilogx(C_values, valid_scores.mean(axis=1), label='test score')

plt.xlabel('C')

plt.legend()
clf = MLPClassifier(solver='lbfgs', random_state=1, activation='logistic', hidden_layer_sizes=(15,))

kfold = KFold(n_splits=5,random_state=7)

cv_results = cross_val_score(clf, x_norm, y, cv=kfold)

print (cv_results.mean()*100, "%")
clf = MLPClassifier(solver='lbfgs', random_state=1, activation='logistic',  hidden_layer_sizes=(15,))

param_grid = {"alpha":10.0 ** -np.arange(-4, 7)}

grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=kfold)

grid.fit(x_norm,y)

print (grid.best_estimator_)

print (grid.best_score_*100, "%")
logreg = linear_model.LogisticRegression(C=0.1)

kfold = KFold(n_splits=5,random_state=7)

cv_results = cross_val_score(logreg, x_norm, y, cv=kfold)

predicted = cross_val_predict(logreg, x_norm, y, cv=kfold)

diff = predicted - y

misclass_indexes = diff[diff != 0].index.tolist()

print (misclass_indexes)
clf = MLPClassifier(solver='lbfgs', random_state=1, activation='logistic', alpha=1.0, hidden_layer_sizes=(15,))

kfold = KFold(n_splits=5,random_state=7)

cv_results = cross_val_score(clf, x_norm, y, cv=kfold)

predicted = cross_val_predict(clf, x_norm, y, cv=kfold)

diff = predicted - y

misclass_indexes = diff[diff != 0].index.tolist()

print (misclass_indexes)
clf1 = linear_model.LogisticRegression(C=0.1)

clf2 = MLPClassifier(solver='lbfgs', alpha=1.0,hidden_layer_sizes=(15,), random_state=1, activation='logistic')

eclf = VotingClassifier(estimators=[('lr', clf1), ('nn', clf2)], voting='soft', weights=[2,1])

cv_results = cross_val_score(eclf, x_norm, y, cv=kfold)

print (cv_results.mean()*100, "%")