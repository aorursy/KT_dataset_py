import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/data-naive/train_data.csv",
        sep=r'\s*,\s*',
        engine='python')

test = pd.read_csv("../input/data-naive/test_features.csv",
        sep=r'\s*,\s*',
        engine='python')
train.shape
d = (train[train['ham'] == True].mean() - train[train['ham'] == False].mean())[train.columns[:54]]
abs(d).sort_values(ascending=False)
d.plot(kind = "bar")
plt.show()
Xtrain = train[train.columns[0:57]]
Ytrain = train["ham"]
i=0
s=0
c=2
n=0
v=[]
while 1==1:
    n = GaussianNB()
    scores = cross_val_score(n, Xtrain, Ytrain, cv=c)
    v.append(scores.mean())
    if scores.mean()>s:
        s=scores.mean()
        i=0
        n=c
    else:
        i+=1
        if i>10:
            break
    c+=1
v
n = GaussianNB()
scores = cross_val_score(n, Xtrain, Ytrain, cv=5)
scores
n.fit(Xtrain,Ytrain)
ntest = test.drop("Id", axis = 1)
Ytest = n.predict(ntest)
prediction = pd.DataFrame(index = test.index)
prediction['ham'] = Ytest
prediction.to_csv('submission.csv',index = True)
prediction