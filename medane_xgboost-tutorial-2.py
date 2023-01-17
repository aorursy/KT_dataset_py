import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from xgboost import XGBClassifier, plot_importance

from sklearn.datasets import load_svmlight_files

from sklearn.metrics import accuracy_score

from matplotlib import pyplot

from sklearn.model_selection import train_test_split,StratifiedKFold, GridSearchCV





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

X_train,y_train,X_test,y_test = load_svmlight_files(("/kaggle/input/agaricus-dataset/agaricus_train.txt","/kaggle/input/agaricus-dataset/agaricus_test.txt"))
print("Train data set contains {0} lines and {1} columns.".format(X_train.shape[0],X_train.shape[1]))

print("Test data set contains {0} lines and {1} columns.".format(X_test.shape[0],X_test.shape[1]))
params = {

    'objective':'binary:logistic',

    'max_depth':2,

    'learning_rate':1,

    'silent':1, #do not show the logs

    'n_estimators':5

}
model = XGBClassifier(**params)
model.fit(X_train,y_train)
predictions = model.predict(X_test)

predictions
accuracy = accuracy_score(y_test,predictions)

print("Accuracy of the model : {:0.6f}".format(accuracy))
plot_importance(model)

pyplot.plot()
n_estimators = [50, 100, 150, 200]

max_depth = [2, 4, 6, 8]

learning_rate = [0.0001,0.001,0.01,0.1,0.2,0.3]

param_grid = dict(learning_rate=learning_rate,max_depth=max_depth, n_estimators=n_estimators)



kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)

grid_search = GridSearchCV(model, param_grid,scoring="precision",n_jobs=-1,cv=kfold)

grid_result = grid_search.fit(X_train,y_train)



print("Best: %f using %s" % (grid_result.best_score_,grid_result.best_params_))
params = {

    'objective':'binary:logistic',

    'max_depth':6,

    'learning_rate':0.0001,

    'silent':1, #do not show the logs

    'n_estimators':50

}
model = XGBClassifier(**params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

predicitions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test,predicitions)

print("Accuracy : %.2f%%" % (accuracy*100))