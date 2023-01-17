import pandas as pd

import numpy as np

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

import math
df = pd.read_csv('../input/attendancemarks/AttendanceMarksSA.csv')

df.head()
x = df['MSE']

y = df['ESE']

sns.scatterplot(x,y)
b0 = 0

b1 = 0

alpha = 0.01

count = 10000

n = float(len(x))
for i in range(count):

    y_bar = b1*x + b0

    b1 = b1 - (alpha/n)*sum(x*(y_bar-y))

    b0 = b0 - (alpha/n)*sum(y_bar-y)

        

print(b0,b1)
y_bar = b1*x + b0



plt.scatter(x,y)

plt.plot([min(x),max(x)],[min(y_bar),max(y_bar)],color='red')  #regression line

plt.show()
import math

def RSE(y_true,y_predict):

    y_true = np.array(y_true)

    y_predict = np.array(y_predict)

    RSS = np.sum(np.square(y_true-y_predict))

    

    rse = math.sqrt(RSS/(len(y_true)-2))

    return rse



rse = RSE(df['ESE'],y_bar)

print(rse)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
X = np.array(df['MSE']).reshape(-1,1)

Y = np.array(df['ESE']).reshape(-1,1)



lr = LinearRegression()

lr.fit(X,Y)



print(lr.coef_)

print(lr.intercept_)



yp = lr.predict(X)

rse = RSE(Y,yp)



print(rse)