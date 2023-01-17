import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
col_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv', names = col_names, header=0) 
print(df.shape)
df.describe().transpose()
df.head()
target_column = ['class'] 
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe().transpose()
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

grid = {'solver': ['lbfgs', 'sgd', 'adam'], 'activation': ['identity', 'logistic', 'tanh', 'relu']}
clf_cv = GridSearchCV(MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(8,8,8)), grid, n_jobs=-1, cv=10)

#mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)

clf_cv.fit(X_train, y_train)

print("GridSearch():\n")
combinazioni = 1
for x in grid.values():
    combinazioni *= len(x)
print('Per l\'applicazione della GridSearch ci sono {} combinazioni'.format(combinazioni))
print("Migliore configurazione: ",clf_cv.best_params_)
best_config_gs = clf_cv.best_params_
print("Accuracy CV:",clf_cv.best_score_)
ppn_cv = clf_cv.best_estimator_
print('Test accuracy: %.3f' % clf_cv.score(X_test, y_test))

mlp = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=(8,8,8), **best_config_gs)

mlp.fit(X_train,y_train)
predict_train = mlp.predict(X_train)
predict_test = mlp.predict(X_test)
# Matrice di confusione e report di classificazione per il Train
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_train,predict_train))
print(classification_report(y_train,predict_train))
# Matrice di confusione e report di classificazione per il Test
print(confusion_matrix(y_test,predict_test))
print(classification_report(y_test,predict_test))
