import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset=pd.read_csv('../input/glass.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,9].values
y=np.resize(y,(len(y),1))

# Encoding categorical data
# Encoding the Dependent Variable
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

#Splitting the dataset into training and testing dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Design Matrix & Some necessary initializations
X_train=np.concatenate((np.ones((len(X_train),1)),X_train),axis=1)
X_test=np.concatenate((np.ones((len(X_test),1)),X_test),axis=1)
theta=np.zeros((len(X_train[0]),6))
iterations=2400
alpha=0.03

#Gradient Descent
def Gradient_Descent(X,y,iterations,alpha,theta):
    m=len(X)
    for j in range(0,6):
        for i in range(0,iterations):
            hx=sigmoid(X@theta[:,j])
            delta=X.T@(hx-y[:,j])
            theta[:,j]=theta[:,j]-(alpha/m)*delta
    return theta

#Sigmoid Function
def sigmoid(z):
    return 1/(1+np.exp(-z))    

#Decoding values y_test which were encoded using OneHotEncoder & calcultaing which class has the maximum probability for each of the test sample
def one_hot_decoder(y):
    Y= np.zeros((len(y),1))
    for i in range(len(y)):
        x=np.argmax(y[i,:])+1
        if x>3:
            x=x+1
        Y[i][0]=x
    return Y

#Calculating predicted values
def cal_y_pred_val(X_test, theta):
    y_pred=np.zeros((len(X_test), len(theta[0])))
    for i in range(len(theta[0])):
        y_pred[:,i]=sigmoid(X_test@theta[:,i])
    return y_pred    

#Training model
theta=Gradient_Descent(X_train,y_train,iterations,alpha,theta)    
print('The theta calculated for the training set by Gradient Descent is as follows:\n')
print(theta)

#Testing model
y_pred=cal_y_pred_val(X_test, theta)
y_pred_decoded=one_hot_decoder(y_pred)
y_test=one_hot_decoder(y_test)
y_pred_decoded=y_pred_decoded.astype(int)
y_test=y_test.astype(int)
print('Predicted values for test data:')
print(y_pred_decoded)
y_comparison=np.concatenate((y_test,y_pred_decoded),axis=1)

print('Comparison between test values & predicted values for test data:')
print(y_comparison)
#Calculating the mean squared error and mean absolute percentage error 
from sklearn.metrics import mean_squared_error
m_squared_errror=mean_squared_error(y_test, y_pred_decoded)

def mean_absolute_percentage_error(y_test, y_pred): 
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

m_absolute_percentage=mean_absolute_percentage_error(y_test, y_pred_decoded)
print('The mean squared error = {0:.2f} \nThe mean absolute percentage error = {1:.2f}%\n'.format(m_squared_errror,m_absolute_percentage))
