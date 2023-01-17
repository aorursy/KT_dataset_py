from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection  import cross_val_score

import matplotlib.pyplot as plt

%matplotlib inline
iris=load_iris()



x=iris.data

y=iris.target
knn=KNeighborsClassifier(n_neighbors=5)

scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')

print (scores)
print (scores.mean())
k_range=range(1,31)

k_score=[]

for k in k_range:

    knn=KNeighborsClassifier(n_neighbors=k)

    scores=cross_val_score(knn,x,y,cv=10,scoring='accuracy')

    k_score.append(scores.mean())

print (k_score)
plt.plot(k_range,k_score)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross Val Accuracy')
from sklearn.model_selection  import GridSearchCV
k_range=range(1,31)

print (k_range)
param_grid=dict(n_neighbors=k_range)

print (param_grid)
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(x,y)
grid.grid_scores_
print (grid.grid_scores_[0].parameters)

print (grid.grid_scores_[0].cv_validation_scores)

print (grid.grid_scores_[0].mean_validation_score)
grid_mean_scores=[result.mean_validation_score for result in grid.grid_scores_]

print (grid_mean_scores)
plt.plot(k_range,grid_mean_scores)

plt.xlabel('Value of K for KNN')

plt.ylabel('Cross Val Accuracy')
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
weight_options=['uniform','distance']
param_grid=dict(n_neighbors=k_range,weights=weight_options)

print (param_grid)
grid=GridSearchCV(knn,param_grid,cv=10,scoring='accuracy')
grid.fit(x,y)
grid.grid_scores_
print (grid.best_score_)

print (grid.best_params_)
knn=KNeighborsClassifier(n_neighbors=13,weights='uniform')

knn.fit(x,y)

new=([3,5,4,2],[5,4,3,2])
knn.predict(new)