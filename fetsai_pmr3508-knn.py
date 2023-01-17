import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape

train.head()
ntrain=train.loc[train['parentesco1'] == 1]

col=ntrain.columns[ntrain.isnull().any()].tolist()
number = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

mtrain = ntrain.select_dtypes(include=number)
Xtrain = mtrain.drop(col ,axis = 'columns')
Ytrain = mtrain.Target
Xtrain = Xtrain.drop("Target" ,axis = 'columns')
Xtrain.head()
ntest=test.loc[test['parentesco1'] == 1]

mtest = ntest.select_dtypes(include=number)
Xtest = mtest.drop(col ,axis = 'columns')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
i=0
s=0
c=1
n=0
while 1==1:
    knn = KNeighborsClassifier(n_neighbors=c)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
    if scores.mean()>s:
        s=scores.mean()
        i=0
        n=c
    else:
        i+=1
        if i>10:
            break
    c+=1
n,s
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(Xtrain,Ytrain)
Ytest = knn.predict(Xtest)
prediction = pd.DataFrame(ntest.Id)
prediction["Target"] = Ytest
prediction
prediction.to_csv("prediction.csv", index=False)