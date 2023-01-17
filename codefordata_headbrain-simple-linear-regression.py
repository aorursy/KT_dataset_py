# Importing Necessary libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/headbrain.csv')

df.head()
df.info()
df.isnull().sum()
df.shape
# Taking x and y variables

X = df['Head Size(cm^3)'].values

Y =  df['Brain Weight(grams)'].values
X.shape
Y.shape
mean_X = np.mean(X)

mean_Y = np.mean(Y)



n = len(X)



num =0

denom = 0



for i in range(n):

    num += (X[i]-mean_X)* (Y[i]-mean_Y)

    denom +=(X[i]-mean_X)**2

m = num/denom

c = mean_Y - (m*mean_X)



print(m,',',c)
plt.scatter(X,Y)
min_x = np.min(X)-100

max_x = np.max(X)+100
x = np.linspace(min_x,max_x,1000)
y = m*x+c
plt.scatter(X,Y,color='g')

plt.plot(x,y,color='r')

plt.title('Simple Linear Regression')

plt.xlabel('Head size cm^3')

plt.ylabel('Brain weight in grams')
sum_pred = 0

sum_act = 0



for i in range(n):

    y_pred = (m*X[i]+c)

    sum_pred += (Y[i]-y_pred)**2

    sum_act +=(Y[i]-mean_Y)**2



r2 = 1-(sum_pred/sum_act)

print(r2)
def predict(x):

    y = m*x + c

    print(y)
predict(4177)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



X  = X.reshape((n,1))
X.shape
y.shape
lg = LinearRegression()
lg.fit(X,Y)
y_pred = lg.predict(X)
mse = mean_squared_error(Y,y_pred)
rmse = np.sqrt(mse)
r2_score = lg.score(X,Y)
print(rmse)

print(r2_score)
lg.predict([[4177]])
lg.intercept_