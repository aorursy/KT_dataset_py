import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split#just to split dataset into training set and validation set

%matplotlib inline
path="../input/weatherww2/Summary of Weather.csv"#filepath for the dataset

data=pd.read_csv(path)#read dataset(csv file) from our filepath into a dataframe

data2=pd.read_csv("../input/weatherww2/Weather Station Locations.csv")
data.head()
print(data.shape)
data.describe()#use a built in function to get all properties 
data.plot(x="MinTemp",y="MaxTemp",style='o')

plt.xlabel('MinTemp')

plt.ylabel('MaxTemp')

plt.show()
df=data.iloc[:,[4,5]]#extract MinTemp and Maxtemp into another dataframe

x=df.iloc[:,0].to_numpy().reshape(-1,1)

y=df.iloc[:,1].to_numpy().reshape(-1,1)
#Just a sanity check here

print(x.shape)

print(y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
#For a sanity check ,print shape of datasets

print("Size of training set:\n",x_train.shape)

print(y_train.shape)

print("Size of testing set:\n",x_test.shape)

print(y_test.shape)
params=np.zeros((2,1))
iterations=60000

learning_rate=0.001
x_train=np.hstack((np.ones_like(x_train),x_train))#simply add a column of 1s in the training set 
def computeCost(x,y,w):

    temp=np.dot(x,w)-y

    return np.sum(np.power(temp,2))/(2*len(y))
J=computeCost(x_train,y_train,params)

print(J)
def gradientDescent(x,y,w,learning_rate,iterations):

    for i in range(iterations):

        temp=np.dot(x,w)-y

        temp=np.dot(x.T,temp)

        w=w-(learning_rate/len(y))*temp

    return w
params=gradientDescent(x_train,y_train,params,learning_rate,iterations)
print(params)
print(computeCost(x_train,y_train,params))
plt.scatter(x_train[:,1],y_train)

plt.plot(x_train,np.dot(x_train,params))

plt.xlabel("Minimum Temperature")

plt.ylabel("Maximum Tempearture")

plt.show()