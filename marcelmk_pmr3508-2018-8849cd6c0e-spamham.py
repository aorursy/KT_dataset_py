import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("../input/featdataset/train_data.csv", index_col = 'Id')
train.head()
plt.figure(figsize=(15,10))
(train[train['ham'] == True].mean() - train[train['ham'] == False].mean())[train.columns[:54]].plot(kind = 'bar')
(train[train['ham'] == True].mean() - train[train['ham'] == False].mean())[train.columns[54:57]].plot(kind = 'bar')
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
Xtrain = train[train.columns[0:57]]
Xtrain.head()
Ytrain = train['ham']
Ytrain.head()
NB = GaussianNB()

scores = cross_val_score(NB, Xtrain,Ytrain, cv=10)
scores
test = pd.read_csv('../input/featdataset/test_features.csv', index_col = 'Id')
test.head()
NB.fit(Xtrain,Ytrain)
Ytest = NB.predict(test)
pred = pd.DataFrame(index = test.index)
pred['ham'] = Ytest
pred.to_csv('submission.csv',index = True)