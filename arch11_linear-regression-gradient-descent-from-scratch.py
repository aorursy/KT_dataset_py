# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Univariate lienar regression

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



display(df.shape)

df.head(3)

df.info()
#setting up X,y



X = df["GrLivArea"]

y = df["SalePrice"]

#Feature scaling

import numpy as np



X = (X-X.mean())/X.std()







#create X,y matrices

#m = training examples

#n = no of features

#X - feature vector of m*n+1 dimensions



rowsX = X.shape[0]



X = np.c_[np.ones(rowsX),X]







#Gradient descent

alpha = 0.01

iterations = 2000

m = y.size

np.random.seed(123)

theta = np.random.rand(2) #initial theta



def gradient_descent(x,y,alpha,theta,iterations):

    costs_log=[]

    thetas_log=[]

    for i in range(m):

        

        prediction = np.dot(x,theta)

        error = prediction - y

        cost = 1/(2*m) *np.dot(error.T,error)

        costs_log.append(cost)

        

        theta = theta - alpha*(1/m)*np.dot(x.T,error)  

        thetas_log.append(theta)

        

        #x is (m,n+1) so x.T will be (n+1,m) and error is (m,1) 

        #result theta will be (n+1,1) theta0, theta1, ......

        

    return thetas_log,costs_log



thetas_log , costs_log = gradient_descent(X,y,alpha=0.01,theta=theta,iterations=2000)

final_theta = thetas_log[-1]     #list of lists



print("theta0: "+str(final_theta[0]))



print("theta1: "+str(final_theta[1]))





    
#plotting cost function

plt.title('Cost Function J')

plt.xlabel('No. of iterations')

plt.ylabel('Cost')

plt.plot(costs_log)



plt.show()
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



Xtest = df_test["GrLivArea"]

Xtest = (Xtest-Xtest.mean())/Xtest.std()

Xtestcopy = Xtest

Xtest = np.c_[np.ones(Xtest.shape[0]),Xtest]



predictions = np.dot(Xtest,final_theta)



plt.scatter(df["GrLivArea"],df["SalePrice"])





plt.plot(Xtestcopy,predictions )