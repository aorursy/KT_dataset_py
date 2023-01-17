import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn import neighbors
from sklearn.model_selection import GridSearchCV

train = pd.read_csv("../input/emnist-balanced-train.csv",delimiter = ',')
test = pd.read_csv("../input/emnist-balanced-test.csv", delimiter = ',')

#print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
del train

test_x = test.iloc[:,1:]
test_y = test.iloc[:,0]
del test
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
img1 = [[0]*28]*28
img1 = np.array(img1)
for i in range(28):
    img1[:,i]=train_x.iloc[1,i*28:i*28+28]

fig,ax = plt.subplots()
im = ax.imshow(img1,cmap='Greys')

n_neighbors = 2
weights = 'uniform'

model = neighbors.KNeighborsClassifier(n_neighbors, n_jobs = -1, weights=weights)
model.fit(train_x,train_y)
print(model.score(test_x,test_y))

parameters = {"n_neighbors": np.arange(1,35,2), "metric": ["euclidean","cityblock"]}

tuned_model = GridSearchCV(model,parameters)
tuned_model.fit(train_x[:10000],train_y[0:10000])
#tuned_model.score(test_x,test_y)

bestparams = tuned_model.best_params_
print(bestparams)

model2 = neighbors.KNeighborsClassifier(n_neighbors = bestparams['n_neighbors'], n_jobs = -1, weights=weights, metric = bestparams['metric'])
model2.fit(train_x,train_y)
print(model2.score(test_x,test_y))