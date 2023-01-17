import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

import os

print("read comments for understanding the flow")

print(os.listdir("../input"))

# reading both training and testing datasets

train=pd.read_csv('../input/training.csv')

test=pd.read_csv('../input/testing.csv')

print(train.shape)

print(test.shape)



# separating dependent variables

y_train=train["class"]

y_test=test["class"]



#dropping dependent variables from initial data set

x_train=train.drop("class",axis=1)

x_test=test.drop("class",axis=1)



# initialising model variable as knn classifier

model = KNeighborsClassifier()

#creating a dictionary for all the values we want to try for n_neighbors

param_grid = {'n_neighbors': np.arange(1, 30)}

#using gridsearch to test all values for n_neighbors

knn_gscv = GridSearchCV(model, param_grid, cv=5)

#fitting model to data

knn_gscv.fit(x_train, y_train)



print("best parameter of n_neighbors",knn_gscv.best_params_)



#using the n_neighbors value with best score to train our data

model = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])

model.fit(x_train,y_train)



#applying our model on test data set and calculating the accuracy

y=model.predict(x_test)

acc=accuracy_score(y_test,y)

print("Accuracy of our model is:",acc*100,"%")


