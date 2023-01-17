import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv")
train.shape
train.head()
train.dtypes.value_counts()
train.select_dtypes(include='object').columns
train2 = train.drop(columns = ['Id', 'idhogar', 'dependency', 'edjefe', 'edjefa'])
null_columns = train2.columns[train2.isnull().any()]
train2[null_columns].isnull().sum()
train3 = train2.drop(columns = ['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned'])
ntrain = train3.dropna()
ntrain.head()
test = pd.read_csv("../input/test.csv")
null_columns = test.columns[test.isnull().any()]
test[null_columns].isnull().sum()
Xtest = test.drop(columns = ['Id', 'idhogar', 'dependency', 'edjefe', 'edjefa', 'v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned'])
Xntest = Xtest.dropna()
Xid = test.drop(columns = ['idhogar', 'dependency', 'edjefe', 'edjefa', 'v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned'])
Xnid = Xid.dropna()

Xid.shape

Xtrain = ntrain.drop(columns = ["Target"])
Ytrain = ntrain.Target
knn = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
np.mean(scores)
knn = KNeighborsClassifier(n_neighbors = 97)
knn.fit(Xtrain, Ytrain)
Ypred = knn.predict(Xntest)
result = np.vstack((Xid["Id"], Ypred)).T
x = ["Id","Target"]
Resultado = pd.DataFrame(columns = x, data = result)
Resultado.to_csv("results.csv", index = False)
Resultado
