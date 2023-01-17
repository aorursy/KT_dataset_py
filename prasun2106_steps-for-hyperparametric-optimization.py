import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('../input/grid-search/data.csv',header = None)
plt.figure()

plt.scatter(df.iloc[:,0],df.iloc[:,1],c = df.iloc[:,2])

plt.title('Scatter plot of x and y')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()
X = df.iloc[:,0:2]

y = df.iloc[:,2]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

y_pred_train = clf.predict(X_train)
from sklearn import metrics
print(f'accuracy = {metrics.accuracy_score(y_pred, y_test)}')

print('\n')

print(f'{metrics.classification_report(y_pred, y_test)}')

print('\n')

print(f'{metrics.confusion_matrix(y_pred, y_test)}')

print('\n')

print(f'f1_score_train = {metrics.f1_score(y_pred_train, y_train)}')

print('\n')

print(f'f1_score = {metrics.f1_score(y_pred, y_test)}')
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf, X, y, cv = 5)

score
print(f'accuracy = {score.mean():0.2f} +/- {score.std()*2:0.2f}')
# cross validation using shuffle split

from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits = 5, test_size = 0.3,random_state = 42)

score_shuffle = cross_val_score(clf, X, y, cv = cv)
print(f'accuracy = {score_shuffle.mean():0.2f} +/- {score_shuffle.std()*2:0.2f}')
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score



#Choose the model

clf = DecisionTreeClassifier()



#Define the parameters

parameters = {'max_depth':np.arange(1,10), 

              'min_samples_split':np.arange(2,10),

               'min_samples_leaf': np.arange(1,10)}



#Make scorer

scorer = make_scorer(f1_score)



#Make Grid Seach Object

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)



#Fit

grid_obj.fit(X_train, y_train)



#Find the best estimator

best_clf = grid_obj.best_estimator_
#Predict using the best clf

y_pred_grid = best_clf.predict(X_test)

y_pred_grid_train = best_clf.predict(X_train)


print(f'training set f1_score without grid search cv ={f1_score(y_pred_train, y_train)}')

print(f'testing set f1_score without grid search cv ={f1_score(y_pred, y_test)}')

print('\n')

print(f'training set f1_score using grid search cv ={f1_score(y_pred_grid_train, y_train)}')

print(f'testing set f1_score using grid search ={f1_score(y_pred_grid, y_test)}')


