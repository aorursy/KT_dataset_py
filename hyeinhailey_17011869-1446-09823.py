import torch

import pandas as pd

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



torch.manual_seed(777)

xy=pd.read_csv('../input/mlregression-cabbage-price/train_cabbage_price.csv')

xtest=pd.read_csv('../input/mlregression-cabbage-price/test_cabbage_price.csv')

submit=pd.read_csv('../input/mlregression-cabbage-price/sample_submit.csv')



xy['year'] = xy['year'] % 10000

xtest['year'] = xtest['year'] % 10000

#xy=xy.drop('rainFall', axis=1)

#xtest=xtest.drop('rainFall', axis=1)



#xy['rainFall']=sc.fit_transform(xy['rainFall'].values.reshape(-1,1))

#xtest['rainFall']=sc.transform(xtest['rainFall'].values.reshape(-1,1))



x=xy.drop('avgPrice', axis=1)

y=xy['avgPrice']







Xtrain, Xtest, Ytrain, Ytest = train_test_split(x,y, random_state=777, test_size=0.3)

print(Xtrain.shape)

print(Xtest.shape)

print(Ytrain.shape)

print(Ytest.shape)

knn=KNeighborsClassifier(n_neighbors=500, weights='distance', p=5)

knn.fit(Xtrain, Ytrain)

#knn.fit(xstd,y)



Ytrainpred=knn.predict(Xtrain)

Ytestpred=knn.predict(Xtest)

print('Misclssified training samples: %d'%(Ytrain != Ytrainpred).sum())

#오분류 데이터 갯수 확인

from sklearn.metrics import accuracy_score

print("%.4f"%(accuracy_score(Ytest, Ytestpred)*100),'%')



guesses=knn.predict(xtest)

#guesses=knn.predict(xteststd)

guesses[100:115]
for i in range(len(guesses)):

  submit['Expected'][i]=guesses[i]



submit.to_csv('submit.csv', mode='w', header=True, index=False)