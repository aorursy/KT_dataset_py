import numpy as np 

import matplotlib.pyplot as plt
np.random.seed(0)

m = 100

X = np.linspace(0,10,m).reshape(m,1)

y = X + np.random.randn(m,1)
plt.scatter(X,y)
from sklearn.linear_model import LinearRegression as LR
model = LR()

model.fit(X,y)

print(' R^2 = ',model.score(X,y))



predictions = model.predict(X)



plt.scatter(X,y)

plt.plot(X,predictions,c='r')
np.random.seed(0)

m = 100

X = np.linspace(0,10,m).reshape(m,1)

y = X**2 + np.random.randn(m,1)



plt.scatter(X,y)
from sklearn.svm import SVR
model = SVR(C=100)

model.fit(X,y)

print(' R^2 = ',model.score(X,y))



predictions = model.predict(X)



plt.scatter(X,y)

plt.plot(X,predictions,c='r',lw=3)



import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train = train [['Survived','Pclass','Sex','Age']]

train.head()
train.dropna(axis=0,inplace=True)

train
train['Sex'].replace(['male','female'],[0,1],inplace=True)

train.head()
from sklearn.neighbors import KNeighborsClassifier as KNC 
model = KNC()



y = train['Survived']

x = train.drop('Survived',axis=1)



model.fit(x,y)

model.score(x,y)
predict = model.predict(x)
def survive ( model ,pclass= 3, sex = 1 , age = 40 ):

    x = np.array([pclass,sex,age]).reshape(1,3)

    print(model.predict(x))

    print(model.predict_proba(x))
survive(model)