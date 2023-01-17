# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Read the training and the testing data

training=pd.read_csv('../input/training.csv')

testing=pd.read_csv('../input/testing.csv')
#Considering b1-b9 as dependant variable

X_train=training.iloc[:,1:10]

y_train=training.iloc[:,0]

X_test=testing.iloc[:,1:10]

y_test=testing.iloc[:,0]
#Initiating KNN with 4(arbitrary). Will shortly use gridsearchCV for tuning the value of n_neighbors

knn = KNeighborsClassifier(n_neighbors=4)
#Initiating SFS with 3(arbitrary). Will shortly use gridsearchCV for tuning the value of k_features

sfs1 = SFS(knn, 

           k_features=3, 

           forward=True, 

           floating=False, 

           verbose=2,

           scoring='accuracy',

           cv=0)



sfs1 = sfs1.fit(X_train, y_train)
# Setting up pipeline to tune K-features and n_neighbours parameter for SFS and KNN respectively

from sklearn.pipeline import Pipeline

pipe = Pipeline([('sfs', sfs1), 

                 ('knn', knn)])
#Declaring all possible values of k_features ans n_neighbors to be tested through gridSearchCV

param_grid = [

  {'sfs__k_features': [1, 2, 3, 4,5,6,7,8,9],

   'sfs__estimator__n_neighbors': [1, 2, 3, 4,5,6,7,8,9]}

  ]
#Declaring gridsearchCV from param grid declared above with a 5 fold cross validation

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(estimator=pipe, 

                  param_grid=param_grid, 

                  scoring='accuracy', 

                  n_jobs=1, 

                  cv=5,  

                  refit=False)
gs = gs.fit(X_train, y_train)
#We found the best tuned parameters for n_neighbors and k_feature is 1 and 5 

gs.best_params_
#Setting refit=True to get the feature name for k_features

gs = GridSearchCV(estimator=pipe, 

                  param_grid=param_grid, 

                  scoring='accuracy', 

                  n_jobs=1, 

                  cv=5, 

                  refit=True)

gs = gs.fit(X_train, y_train)
#Fetching the indices of the best k_features

gs.best_estimator_.steps[0][1].k_feature_idx_



#Interestingly, Red spectral information is much relevant during the summer months of March and May, Green spectral information has an impact during the after monsoon time of September.
#The best score was found to be around 94%

gs.best_score_
#Creating the training and testing set with the best feature identified through SFS

X_train_sfs=X_train.iloc[:,[0,2,4,5,8]]

y_train_sfs=y_train

X_test_sfs=X_test.iloc[:,[0,2,4,5,8]]

y_test_sfs=y_test
#Implementing KNN with n_neighbour=3. The n_neighbor=3 is chosen applying the elbow method

classifier=KNeighborsClassifier(n_neighbors=3) 

classifier.fit(X_train_sfs, y_train_sfs) 
#Predicting the class for the test data

y_pred = classifier.predict(X_test_sfs)  
#We achieve a score of around 83.3%

accuracy_score(y_test_sfs, y_pred)