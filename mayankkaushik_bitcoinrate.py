import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

d=pd.read_csv("../input/bitcoinmain.csv")
d.head()
d.shape
d.dtypes
d['Date'] = pd.to_datetime(d['Date'])
d.dtypes
data=d.set_index("Date")
d.corr()
data.head()
data['avg']=data[['Open*','Close**']].mean(axis=1)
data.head()
data.drop(columns=['High','Low','Volume','Market Cap'], inplace = True)

data.head()
x=data['Open*']

y=data['Close**']

plt.figure(figsize=(15,8))

plt.plot(x,color='r')

plt.plot(y,color='b')

plt.show
plt.figure(figsize=(15,8))

plt.plot(data.index,data.avg)
from sklearn.model_selection import train_test_split
x=data['Open*']

y=data['Close**']
x.head()
y.head()
Y = data['Close**']

X=data.drop(columns=['Close**','avg'])
xtrain = X['2018-6':'2013']

ytrain = Y['2018-6':'2013']

xtest = X['2019':'2018-7']

ytest = Y['2019':'2018-7']
ytrain.shape
xtest.shape


from sklearn.linear_model import LinearRegression

clf=LinearRegression()
train=np.array(xtrain).reshape(-1,1)

xtest=np.array(xtest).reshape(-1,1)
clf.fit(xtrain,ytrain)
clf.score(xtest,ytest)
clf.score(xtrain,ytrain)
plt.figure(figsize=(12,7))

ytrain.plot()

ytest.plot()
clf.coef_
clf.intercept_
clf.predict(xtest)