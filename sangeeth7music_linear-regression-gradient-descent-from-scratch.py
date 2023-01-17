import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
cols=['bedrooms','price']

df=pd.read_csv("../input/kc_house_data.csv",usecols=cols)

df.head()
x=df['bedrooms']

y=df['price']
# standardizing the input values

def standardize(x):

    return (x-np.mean(x))/np.std(x)
X=standardize(x)

X=np.c_[np.ones(x.shape[0]),X]
alpha=0.01

m=y.size

np.random.seed(23)

theta=np.random.rand(2)

iterations=2000



def gradient_descent(x,y,theta,alpha,iterations):

    previous_costs=[]

    previous_thetas=[theta]

    for i in range(iterations):

        prediction=np.dot(x,theta) #line equation (theta*x)

        error=prediction-y # error value

        cost=1/(2*m)*np.dot(error.T,error) #cost function

        previous_costs.append(cost)

        theta=theta-(alpha*(1/m)*np.dot(x.T,error)) #updating theta values

        previous_thetas.append(theta)

    return previous_costs,previous_thetas

costs,thetas=gradient_descent(X,y,theta,alpha,iterations)
plt.title('Cost Function')

plt.xlabel('# of iterations')

plt.ylabel('Cost')

plt.plot(costs)

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=23)

lm=LinearRegression().fit(x_train,y_train)

predictions=lm.predict(x_test)
print("Linear Regression model Intercept:",lm.intercept_)

print("Linear Regression model Theta1",lm.coef_[1])