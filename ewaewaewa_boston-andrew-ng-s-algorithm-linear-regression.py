import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from pandas import read_csv
filename = ("../input/housing.csv")

names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

data = read_csv(filename, delim_whitespace=True, names=names)
print(data.shape)
data.head()
plt.figure(figsize=(12,10))

sns.heatmap(data.corr().round(2),cmap='coolwarm',annot=True)
boston = pd.DataFrame(np.c_[data['LSTAT'], data['RM'], data['MEDV']], columns = ['LSTAT','RM','MEDV'])
boston.head()
def featureNormalization(X):



    mean=np.mean(X,axis=0) 

    std=np.std(X,axis=0)

    

    X_norm = (X - mean)/std

    

    return X_norm , mean , std



boston_n=boston.values

m=len(boston_n[:,-1])

X=boston_n[:,0:2].reshape(m,2)

X, mean_X, std_X = featureNormalization(X)

X = np.append(np.ones((m,1)),X,axis=1)

y=boston_n[:,-1].reshape(m,1)

theta=np.zeros((3,1))
def computeCost(X,y,theta):

    

    m=len(y)

    predictions=X.dot(theta)

    square_err=(predictions - y)**2

    

    return 1/(2*m) * np.sum(square_err)
computeCost(X,y,theta)
def gradientDescent(X,y,theta,alpha,num_iters):

     

    m=len(y)

    J_history=[]

    

    for i in range(num_iters):

        predictions = X.dot(theta)

        error = np.dot(X.transpose(),(predictions -y))

        descent=alpha * 1/m * error

        theta-=descent

        J_history.append(computeCost(X,y,theta))

    

    return theta, J_history
theta,J_history = gradientDescent(X,y,theta,0.01,300)

print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1 + "+str(round(theta[2,0],2))+"x2");
theta
plt.plot(J_history)

plt.xlabel("Iteration")

plt.ylabel("$J(\Theta)$")

plt.title("Cost function using Gradient Descent")
computeCost(X,y,theta)
predictions = X.dot(theta)
# sum of square of residuals

ssr = np.sum((predictions - y)**2)



#  total sum of squares

sst = np.sum((y - np.mean(y))**2)



# R2 score

r2_score = 1 - (ssr/sst)
print(' R2 score =' +str(round(r2_score,3)))
# mean squared error

mse = np.sum((predictions - y)**2)



# root mean squared error

# m is the number of training examples

rmse = np.sqrt(mse/m)
print('MSE = ', mse, 'RMSE = ', rmse)
def predict(x,theta):

    

    predictions= np.dot(theta.transpose(),x)

    

    return predictions[0]
x_new1=np.array([8.25,5.50])

x_new1=np.append(np.ones(1),x_new1)

predict_new1=predict(x_new1,theta)



print(predict_new1)