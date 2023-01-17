import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("../input/dataset/train_data.csv", index_col='Id')
train
train.describe()
(train == '?').any()
train[train == '?'].count()
train['workclass'].value_counts().plot(kind = 'bar')
train['occupation'].value_counts().plot(kind = 'bar')
train['native.country'].value_counts().plot(kind = 'bar')
train['workclass'] = train['workclass'].replace(to_replace = '?', value = 'Private')
train['native.country'] = train['native.country'].replace(to_replace = '?', value = 'United-States')
train[train == '?'].count()
train = train.loc[train.occupation != '?']
train
Xtrain = train.loc[:,'age':'native.country']
Ytrain = train.income
Xtrain.head()
Ytrain.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
Xtrain = pd.get_dummies(Xtrain)
Xtrain
knn = KNeighborsClassifier(n_neighbors = 10)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores.mean()
scores_array = []
for i in range(1,25):
    knn = KNeighborsClassifier(n_neighbors = i)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    scores_array.append(scores.mean())
    

plt.plot(scores_array, 'ro')
test = pd.read_csv('../input/dataset/test_data.csv', na_values='?', index_col='Id')
test
Xtest = pd.get_dummies(test)
Xtest.head()
##garantindo que ambas tenham a mesma dimensão
missing_cols = set( Xtrain.columns ) - set( Xtest.columns )
for c in missing_cols:
    Xtest[c] = 0
Xtest = Xtest[Xtrain.columns]
knn = knn = KNeighborsClassifier(n_neighbors = 21)
knn.fit(Xtrain,Ytrain)
YtestPred = knn.predict(Xtest)
YtestPred
prediction = pd.DataFrame(index = test.index)
prediction['income'] = YtestPred
prediction
prediction.to_csv("submition.csv")