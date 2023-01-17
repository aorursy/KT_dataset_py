import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv('../input/predictingese/AttendanceMarksSA.csv')

data.head()

X= data['MSE']

Y=data['ESE']

sns.scatterplot(X,Y)
theta0 = 0

theta1 = 0

alpha = 0.01

count = 10000

m = len(X) # m is number of examples i.e number of students here.
for i in range(count): 

    Y_hat = theta1*X + theta0  

    theta0 = theta0 - (alpha/m)*sum(Y_hat-Y)

    theta1 = theta1 - (alpha/m)*sum(X*(Y_hat-Y))

    

    

print(theta0,theta1)
Y_hat = theta1*X + theta0



plt.scatter(X, Y) 

plt.plot([min(X), max(X)], [min(Y_hat), max(Y_hat)], color='red')  # regression line

plt.show()
import math

def RSE(y_true, y_predicted):

   

    y_true = np.array(y_true)

    y_predicted = np.array(y_predicted)

    RSS = np.sum(np.square(y_true - y_predicted))



    rse = math.sqrt(RSS / (len(y_true) - 2))

    return rse





rse= RSE(data['ESE'],Y_hat)

print(rse)


from sklearn.linear_model import LinearRegression
X = np.array(data['MSE']).reshape(-1,1)

y = np.array(data['ESE']).reshape(-1,1)

 



model = LinearRegression()

model.fit(X,y)





print(model.coef_)

print(model.intercept_)



y_predict = model.predict(X)



rse = RSE(y,y_predict)



print(rse)
marks = [17]

result = model.predict([marks])
print(result)