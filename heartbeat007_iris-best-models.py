import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('../input/Iris.csv')
df.head()
df = df.drop('Id',axis=1)
df.head()
## hot encoding 

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

result = df.Species

df['class'] = encoder.fit_transform(df['Species'])
df1=df.drop('Species',axis=1)
df1.head()
df1.corr()['class'].plot()
X = df.drop(['Species'], axis=1)

y = df['class']
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV,KFold,StratifiedKFold
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC,LinearSVC

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
k_range = list(range(1,31))

weight_options = ["uniform", "distance"]



param_grid = dict(n_neighbors = k_range, weights = weight_options)

print (param_grid)

knn = KNeighborsClassifier()



grid = GridSearchCV(knn, param_grid, cv = 10,n_jobs=-1,verbose=2 ,scoring = 'accuracy')

grid.fit(x_train,y_train)
print (grid.best_estimator_)
knn_final = grid.best_estimator_
from sklearn.model_selection import GridSearchCV



# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}



# Create a based model

rf = RandomForestClassifier()



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2) ## njobs force the computer to use all their resources
grid_search.fit(x_train, y_train)
print (grid_search.best_estimator_)
rf_final=grid_search.best_estimator_
param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
grid2 = GridSearchCV(SVC(),param_grid,n_jobs=-1,refit = True, verbose=2)
grid2.fit(x_train,y_train)
print (grid2.best_estimator_)
svc_final = grid2.best_estimator_
param_grid = {'C':[1,10,100,1000]}

grid3 = GridSearchCV(LinearSVC(),param_grid,refit = True, verbose=2,n_jobs=-1)
grid3.fit(x_train,y_train)
print (grid3.best_estimator_)
lsvc_final = grid3.best_estimator_

parameters = {'learning_rate': [0.1, 0.05, 0.02, 0.01],

              'max_depth': [4, 6, 8],

              'min_samples_leaf': [20, 50,100,150],

              #'max_features': [1.0, 0.3, 0.1] 

              }



grid4 = GridSearchCV(GradientBoostingClassifier(), parameters,verbose=2, cv=10, n_jobs=-1)



grid4.fit(x_train, y_train)

print (grid4.best_estimator_)
gb_final = grid4.best_estimator_
# Define the parameter values that should be searched

sample_split_range = list(range(2, 50))



# Create a parameter grid: map the parameter names to the values that should be searched

# Simply a python dictionary

# Key: parameter name

# Value: list of values that should be searched for that parameter

# Single key-value pair for param_grid

param_grid = dict(min_samples_split=sample_split_range)

dtc = DecisionTreeClassifier()

# instantiate the grid

grid5 = GridSearchCV(dtc, param_grid, cv=10,n_jobs=-1,verbose=2, scoring='accuracy')



# fit the grid with data

grid5.fit(x_train, y_train)
dt_final = grid5.best_estimator_
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=[('knn',knn_final),('rf',rf_final),('dt',dt_final),('svc',svc_final),('lsvc',lsvc_final),('gb',gb_final)],voting='hard')
voting_clf.fit(x_train,y_train)
predicted=voting_clf.predict(x_test)
print(cross_val_score(voting_clf, x_test, y_test, cv=10))