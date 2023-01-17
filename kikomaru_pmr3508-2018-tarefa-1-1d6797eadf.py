import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv("../input/dataset/train_data.csv",
                   na_values = '?')
train.shape
train = train.dropna()
train.shape
Xtrain = train[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
ytrain = train.income
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
scores = []
for k in range(10, 51, 5):
    knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
    scores.append(cross_val_score(knn, Xtrain, ytrain, cv=10))
means = []
k_neighbors = []
for i in range(len(scores)):
    x = scores[i][1]
    k_neighbors.append(10+i*5)
    for q in range(9):
        x += scores[i][q]
    x = x/10
    means.append(x)
means
k_neighbors
plt.plot(k_neighbors, means, 'ro')
plt.ylabel('Accuracy')
plt.xlabel('k_neighbors')
scores = []
for k in range(35, 46, 2):
    knn = KNeighborsClassifier(n_neighbors=k, weights = 'distance')
    scores.append(cross_val_score(knn, Xtrain, ytrain, cv=10))
means = []
k_neighbors = []
for i in range(len(scores)):
    x = scores[i][1]
    k_neighbors.append(35+i*2)
    for q in range(9):
        x += scores[i][q]
    x = x/10
    means.append(x)
knn = KNeighborsClassifier(n_neighbors=40, weights='distance')
k_40 = cross_val_score(knn, Xtrain, ytrain, cv=10)
x = k_40[0]
for i in range(1, len(k_40)):
    x += k_40[i]
x = x/10
means.append(x)
k_neighbors.append(40)

plt.plot(k_neighbors, means, 'ro')
plt.ylabel('Accuracy')
plt.xlabel('k_neighbors')
knn.fit(Xtrain, ytrain)
test = pd.read_csv('../input/dataset/test_data.csv',
                  na_values = '?',
                  index_col = 'Id')
test
test['income'] = np.nan
test
Xtest = test[["age","education.num","capital.gain", "capital.loss", "hours.per.week"]]
ytest = test.income
ytestpred = knn.predict(Xtest)
ytestpred
prediction = pd.DataFrame(index = test.index)
prediction['income'] = ytestpred
prediction
prediction.to_csv("submition.csv")