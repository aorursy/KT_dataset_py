# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# import data



from sklearn.datasets import load_iris

import pandas as pd

import numpy as np

iris = load_iris()

x = iris.data

y = iris.target
# normalization

x = (x-np.min(x))/(np.max(x)-np.min(x))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

# Split Data (Train - Test)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
# K Fold

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X = x_train, y = y_train, cv = 10)

accuracies

print("Average Accuracy : ", np.mean(accuracies))

print("Average std : ", np.std(accuracies))



knn.fit(x_train,y_train)



y_head = knn.predict(x_test)



print("Test accuracy : ",knn.score(x_test,y_test))
# Grid Search 



import numpy as np



from sklearn.model_selection import GridSearchCV



grid = {"n_neighbors" : np.arange(1,50)}

knn = KNeighborsClassifier()



knn_cv = GridSearchCV(knn, grid, cv = 10)

knn_cv.fit(x,y)



print("Tuned Hyperparameter K :  ",knn_cv.best_params_)

print("Best Score : ",knn_cv.best_score_)
# Grid Search CV with Logistic Regression



x = x[:100,:]

y = y[:100]



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)



from sklearn.linear_model import LogisticRegression

 

grid = {"C" : np.logspace(-3,3,7),"penalty" : ["l1","l2"]}



logreg = LogisticRegression(solver='liblinear')

logreg_cv = GridSearchCV(logreg,grid,cv=10)

logreg_cv.fit(x_train,y_train)



print("Best Params : ", logreg_cv.best_params_)

print("Best Score : ",logreg_cv.best_score_)







logreg2 = LogisticRegression(solver='liblinear',C = 0.1 , penalty ="l2")

logreg2.fit(x_train,y_train)

print("Score : ", logreg2.score(x_test,y_test))