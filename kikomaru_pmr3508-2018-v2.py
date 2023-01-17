import pandas as pd
import numpy as np
train = pd.read_csv("../input/train.csv",
                   na_values = "?")
train
train.shape
train.isnull().sum()
ytrain = train.Target
Xtrain = train.drop(train.columns[train.isnull().any()].tolist(), axis=1)
Xtrain = Xtrain.drop(['Id', 'Target'], axis=1)
Xtrain = Xtrain.select_dtypes(exclude=['object'])
Xtrain.columns.tolist()
Xtrain
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
ks = []
scores = []
for k in range(10, 201, 10):
    knn = KNeighborsClassifier(n_neighbors=k)
    ks.append(k)
    scores.append(cross_val_score(knn, Xtrain, ytrain, cv=10).mean())
scores, ks
import matplotlib.pyplot as plt 
plt.plot(ks, scores, 'ro')
plt.ylabel('Score')
plt.xlabel('k')
ks = []
scores = []
for k in range(70, 101, 5):
    knn = KNeighborsClassifier(n_neighbors=k)
    ks.append(k)
    scores.append(cross_val_score(knn, Xtrain, ytrain, cv=10).mean())
scores, ks
plt.plot(ks, scores, 'ro')
plt.ylabel('Score')
plt.xlabel('k')
ks = []
scores = []
for k in range(75, 85, 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    ks.append(k)
    scores.append(cross_val_score(knn, Xtrain, ytrain, cv=10).mean())
plt.plot(ks, scores, 'ro')
plt.ylabel('Score')
plt.xlabel('k')
knn = KNeighborsClassifier(n_neighbors=79)
knn.fit(Xtrain,ytrain)
test = pd.read_csv('../input/test.csv', na_values='?')
test
test['Target'] = np.nan
test.index
test.shape
Xtest = pd.DataFrame(data = test[Xtrain.columns])
ytestpred = knn.predict(Xtest)
ytestpred
prediction = pd.DataFrame()
prediction['Id'] = test.Id
prediction['Target'] = ytestpred
prediction
prediction.to_csv('submition.csv', index = False)
