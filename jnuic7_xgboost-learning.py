# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from xgboost import plot_importance

from sklearn.metrics import accuracy_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/diabetes.csv')

print(dataset.head())

dataset.isnull().sum()
#since no nulls lets fit a model to it and find the most important features

data = dataset.values

X = data[:,0:8]

y = data[:,8]

from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=7)

model = XGBClassifier()

model.fit(X_train,y_train)

print(dataset.columns)

#Built in importance plot

plot_importance(model)
#Lets check the acc of the model



print(X_train.shape)

pred = model.predict(X_test)

pred = [round(value) for value in pred]

acc = accuracy_score(y_test,pred)

print(acc * 100)
#Lets try grid search to tune the learning rate

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



model = XGBClassifier()

lr = [0.0001,0.001,0.01,0.1,0.2,0.3]

n_estim = [100,200,300,400,500,600]

grid = dict(learning_rate=lr,n_estimators=n_estim)

kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=7)

grid_search = GridSearchCV(model,grid,scoring="neg_log_loss",n_jobs=-1,cv=kfold)

grid_res = grid_search.fit(X_train,y_train)

print("Best: %f using %s" % (grid_res.best_score_, grid_res.best_params_))

means = grid_res.cv_results_['mean_test_score']

stds = grid_res.cv_results_['std_test_score']

params = grid_res.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

	print("%f (%f) with: %r" % (mean, stdev, param))