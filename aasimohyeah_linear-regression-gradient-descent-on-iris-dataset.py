import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('../input/Iris.csv')

#taking only first 50 columns of Iris-setosa species
m,n=df.shape
X=df.SepalLengthCm.iloc[0:50].values.reshape(50,1)
Y=df.SepalWidthCm.iloc[0:50].values.reshape(50,1)

Xnew=np.hstack((np.ones((50,1)),X))

np.random.seed(0)
theta=np.random.randn(1,2)
print('Theta:',theta)

iters=10000
J=np.zeros(iters)
learning_rate=0.001

#training

for i in range(iters):
    J[i]=(1/(2*m))*np.sum((np.dot(Xnew,theta.T)-Y)**2)
    theta[0,0]-=(learning_rate/m)*np.sum(np.dot(Xnew,theta.T)-Y)
    theta[0,1]-=(learning_rate/m)*np.sum((np.dot(Xnew,theta.T)-Y)*X)
plt.plot(np.arange(iters),J)
plt.xlabel('Number of iterations',color='red')
plt.ylabel('Cost',color='red')
plt.title('Cost vs iterations',color='green')
plt.show()
print('Updated theta after Gradient Descent:',theta)
plt.scatter(X,Y,marker='.')
plt.plot(X,np.dot(Xnew,theta.T))
plt.xlabel('Sepal Length(in cm)')
plt.ylabel('Sepal Width(in cm)')
plt.title('Sepal length vs width of Iris-Setosa Species',color='green')
plt.show()