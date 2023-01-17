import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/attmsa/AttendanceMarksSA-200919-184800.csv')

df.head()
x= df['MSE']

y=df['ESE']

sns.scatterplot(x,y)
beta0=0

beta1=0

alpha=0.01

count =10000

n=float(len(x))

for i in range(count): 

    ybar = beta1*x + beta0    

    beta1 = beta1 - (alpha/n)*sum(x*(ybar-y))

    beta0 = beta0 - (alpha/n)*sum(ybar-y)

    

print(beta0,beta1)
ybar = beta1*x + beta0



plt.scatter(x, y) 

plt.plot([min(x), max(x)], [min(ybar), max(ybar)], color='green') 

plt.show()
import math

def RSE(y_true, y_predicted):

   

    y_true = np.array(y_true)

    y_predicted = np.array(y_predicted)

    RSS = np.sum(np.square(y_true - y_predicted))



    rse = math.sqrt(RSS / (len(y_true) - 2))

    return rse





rse= RSE(df['ESE'],ybar)

print(rse)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
x = np.array(df['MSE']).reshape(-1,1)

y = np.array(df['ESE']).reshape(-1,1)

 



lr = LinearRegression()

lr.fit(x,y)





print(lr.coef_)

print(lr.intercept_)



yp = lr.predict(x)

rse = RSE(y,yp)



print(rse)