import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge
train = pd.read_csv("../input/train.csv",
        sep=r'\s*,\s*',
        engine='python')

test = pd.read_csv("../input/test.csv",
        sep=r'\s*,\s*',
        engine='python')
train.head()
train.iloc[:,1:9].hist(figsize =(10,10),bins=50)
plt.show()
for i in train.iloc[:,1:9]:
    plt.scatter(train[i], train['median_house_value'])
    plt.xlabel(i)
    plt.ylabel('median_house_value')
    plt.show()
Xtrain = train[['median_income','total_bedrooms','households','total_rooms']]
Ytrain = train[['median_house_value']]
i=0
s=0
c=1
n=0
v=[]
while 1==1:
    knn = KNeighborsRegressor(n_neighbors=c)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
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

n,s,v
knn = KNeighborsRegressor(n_neighbors=n)
knn.fit(Xtrain,Ytrain)
Xtest = test[['median_income','total_bedrooms','households','total_rooms']]
Ytest = knn.predict(Xtest)
prediction = pd.DataFrame(test.Id)
prediction["median_house_value"] = Ytest
prediction.to_csv("predictionKNN.csv", index = False)
lasso = Lasso()
lasso.fit(Xtrain,Ytrain)
scores = cross_val_score(lasso,Xtrain,Ytrain,cv=10)
scores.mean()
Xtest = test[['median_income','total_bedrooms','households','total_rooms']]
Ytest = lasso.predict(Xtest)
prediction = pd.DataFrame(test.Id)
prediction["median_house_value"] = Ytest
prediction.to_csv("predictionLASSO.csv", index = False)
ridge = Ridge()
ridge.fit(Xtrain,Ytrain)
scores = cross_val_score(ridge,Xtrain,Ytrain,cv=10)
scores.mean()
Xtest = test[['median_income','total_bedrooms','households','total_rooms']]
Ytest = ridge.predict(Xtest)
prediction = pd.DataFrame(test.Id)
prediction["median_house_value"] = Ytest
prediction.to_csv("predictionRIDGE.csv", index = False)