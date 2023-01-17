# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
heart = pd.read_csv('../input/heart.csv')

heart.head()
heart.info()
part1 = heart.iloc[:,:2]

part2 = heart.iloc[:,3:5]

data = pd.concat([part1,part2],axis=1)

data
# heart disease is exist or not?

# 1: disease exist, 0 : not exist

target = heart.iloc[:,-1:]

target
# splitting the data set as train and test set.



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.30, random_state=0)
# Scaling



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# SVM



from sklearn.svm import SVC



classifier = SVC(kernel = 'rbf', random_state=0)

classifier.fit(X_train, y_train)
# Prediction



y_pred = classifier.predict(X_test)

y_pred
# Confusion matrix



from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

cm
from sklearn.model_selection import cross_val_score 



success = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 4)



print(success.mean())



print(success.std()) # the smaller std is the better
# Optimization of parameters and selection of algorithm



from sklearn.model_selection import GridSearchCV



# parameters

p = [{'C':[1,2,3,4,5],'kernel':['linear', 'poly']}, {'C': [0.1,0.25,0.5,0.7,1], 'kernel': ['rbf'], 'gamma':[1,0.5,0.1,0.01,0.05]}]



# estimator : optimize SVM algorithm

# scoring : will be scored according to what 

# cv : how many fold we will use

gs = GridSearchCV(estimator = classifier, param_grid = p, scoring = 'accuracy', cv = 10, n_jobs = -1)



grid_search = gs.fit(X_train, y_train)

grid_search
best_result = grid_search.best_score_

best_result
best_parameters = grid_search.best_params_

best_parameters