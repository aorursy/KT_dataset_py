import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
Xtrain = train.drop(columns=["Id","median_house_value"])
Ytrain = train["median_house_value"]

Xtrain.describe()
tab = Xtrain.corr(method="spearman")
tab
POStrain = train[["latitude","longitude"]]

plt.subplots
plt.xlim(-150,-100)
plt.plot(train["longitude"],train["latitude"],"g.")
plt.show()

plt.subplots
plt.xlim(-150,-100)
plt.scatter(train["longitude"],train["latitude"], c= train["median_house_value"] , cmap = "jet" , s = 10)
plt.show()
from sklearn.model_selection import cross_val_score as cvs
from sklearn.neighbors import KNeighborsRegressor as knr

features = ["median_age", "total_rooms", "total_bedrooms", "median_income"]
Xnewtrain = train[features]

knn_scores = []
for i in range(1,100,5):
    knn = knr(n_neighbors = i)
    scores = cvs(knn, Xnewtrain, Ytrain, scoring='neg_mean_squared_error',
                cv = 5)
    knn_scores.append([i, -scores.mean()])
knn_scores = np.array(knn_scores)
knn_scores
plt.plot(knn_scores[:,0], knn_scores[:,1])
knn_scores[np.where(knn_scores[:,1] == np.amin(knn_scores[:,1]))[0]]
from sklearn.linear_model import Ridge

r = Ridge(alpha=1.0)
scores = cvs(r, Xnewtrain, Ytrain,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
from sklearn.linear_model import Lasso

l = Lasso(alpha=1.0)
scores = cvs(l, Xnewtrain, Ytrain,
             scoring='neg_mean_squared_error',cv = 5)
print(-scores.mean())
r = Ridge(alpha=1.0)

features = ["median_age", "total_rooms", "total_bedrooms", "median_income"]

Xtest = test[features]
ID_list = test.Id.tolist()

r.fit(Xnewtrain, Ytrain)
YPredict = r.predict(Xtest)
from sklearn.metrics import mean_squared_log_error
YPredict = pd.Series(YPredict)
for i in range(YPredict.shape[0]-1):
    if(YPredict[i] < 0):
        YPredict[i] = 0
YPredict.describe()
YPredict
pd.DataFrame({"Id":ID_list,"median_house_value":YPredict}).to_csv("pred_R.csv",index=False)