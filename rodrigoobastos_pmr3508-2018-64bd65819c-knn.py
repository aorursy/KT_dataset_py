import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.head()
train["Target"].value_counts()/(train["Target"].value_counts().sum())
train.shape
train.isnull().sum()[train.isnull().sum()>0]
train.isnull().sum()[train.isnull().sum()>0]/(train["Target"].value_counts().sum())
train2 = train.drop(["v2a1", "v18q1", "rez_esc"], axis=1)
train2.isnull().sum()[train.isnull().sum()>0]
ntrain = train2.dropna();
ntrain
numTrain = ntrain.apply(preprocessing.LabelEncoder().fit_transform)
Xtrain = numTrain.iloc[:,1:139]
Ytrain = numTrain.Target
mscores = []
mscores2 = []
mscores3 = []
for x in range (1, 11):
    knn = KNeighborsClassifier(n_neighbors=(x*100))
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    mscores.append(scores.mean())
for x in range (1, 11):
    knn = KNeighborsClassifier(n_neighbors=(x*10 + mscores.index(max(mscores))*100))
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    mscores2.append(scores.mean())
for x in range (1, 11):
    knn = KNeighborsClassifier(n_neighbors=(x + mscores2.index(max(mscores2))*10))
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    mscores3.append(scores.mean())
max(mscores3)
mscores.index(max(mscores))*100 + mscores2.index(max(mscores2))*10 + mscores3.index(max(mscores3)) + 1
test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test2 = test.drop(["v2a1", "v18q1", "rez_esc"], axis=1)
ntest = test2.dropna()
numTest = ntest.apply(preprocessing.LabelEncoder().fit_transform)
Xtest= numTest.iloc[:,1:139]
knn.fit(Xtrain,Ytrain)
Ytest = knn.predict(Xtest)
Ytest
Ytabela = pd.DataFrame(index=ntest.Id,columns=['income'])
Ytabela['income'] = Ytest
Ytabela
Ytabela.to_csv('prediction.csv')