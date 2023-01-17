import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV


Dataset= pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

Dataframe=pd.DataFrame(Dataset) #get dataframe from csv

iris_flower=Dataframe.values #get features and targets



X = iris_flower[:,:-1] #features

y = iris_flower[:,len(iris_flower[0])-1] #targets





Dataframe.sample(10)

knn2=KNeighborsClassifier()

param_grid={'n_neighbors':np.arange(1,50)}

knn_gscv=GridSearchCV(knn2,param_grid,cv=5)

knn_gscv.fit(X,y)

k=knn_gscv.best_params_['n_neighbors']

print(knn_gscv.best_params_)
knn=KNeighborsClassifier(n_neighbors=k)

knn.fit(X,y)

result=knn.predict([[0.1,0.8,0.2,2.1]])

print(result)


