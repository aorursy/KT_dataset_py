import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/database/train.csv",index_col=0,na_values=' ')
train.head()
train.shape
train['Target'].value_counts().plot(kind='bar')
train['Target'].value_counts()
xtrain=train[['v2a1','hacdor','rooms','hacapo','v14a','refrig','r4h1','r4m1','tamhog','escolari','paredzocalo','paredmad','pisonotiene','planpri']]
ytrain=train.Target
# teste com atributos escolhidos ao acaso
x = train[["refrig","escolari","computer","television","rooms","hhsize","paredblolad","mobilephone","qmobilephone","overcrowding"]]
knn=KNeighborsClassifier(n_neighbors=35)
scores = cross_val_score(knn,x,ytrain,cv=5)
scores.mean()
test=pd.read_csv("../input/database/test.csv", index_col=0,na_values=' ')
test=test.drop(columns=['v2a1','v18q1','rez_esc','edjefe','edjefa','dependency','idhogar'],axis=1)
knn.fit(x,ytrain)
# prediction=knn.predict(test)

# novo teste sem as colunas que tem muitos NaNs e termos não numericos
test=test.dropna()
test.shape
train=train.drop(columns=['v2a1','v18q1','rez_esc','edjefe','edjefa','dependency','idhogar'],axis=1)
train=train.dropna()
ytrain=train.Target
train=train.drop(columns=['Target'])
knn=KNeighborsClassifier(n_neighbors=35)
scores=cross_val_score(knn,train,ytrain,cv=5)
scores.mean()   #desempenho menor do que o teste anterior
knn.fit(train,ytrain)
prediction=knn.predict(test)
prediction
#prediction.to_csv("cr.csv", index=False)
