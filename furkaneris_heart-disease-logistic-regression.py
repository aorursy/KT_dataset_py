# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
heart = pd.read_csv("../input/heart.csv")

heart.head()
heart.info()
plt.figure(figsize = (12,10))

sns.heatmap(heart.corr(),annot = True)

plt.show()
x = heart.drop("target",axis = 1)

y = heart.target.values
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

print("x_train : ",x_train.shape)

print("x_test : ",x_test.shape)

print("y_train : ",y_train.shape)

print("y_test : ",y_test.shape)
class LogReg():

    def forward_backward_propagation(self,w,b,x_train,y_train):

        x_train = x_train.T #(13,242)

        y_train = y_train.T #(242,)

        w = w.T     # if we want to multiply two matrix,first matrix column number and second matrix row number must be equal.

        z = np.dot(w,x_train)+b # This is our model

        y_head = 1/(1+np.exp(-z)) # Sigmuid function

        loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) 

        cost = np.sum(loss)/x_train.shape[1] # cost function

        

        dw = np.dot(x_train,(y_head-y_train).T)/x_train.shape[1] #derivative weight

        db = np.sum(y_head-y_train)/x_train.shape[1] #derivative bias

        

        gradients = {"dw":dw,"db":db}

        

        return cost,gradients

    def fit(self,w,b,x_train,y_train,learn_rate,num_it):

        #To reach most fit model we should update our model

        self.cost_list = []

        for i in range(num_it):

            cost,gradients = self.forward_backward_propagation(w,b,x_train,y_train)

            self.cost_list.append(cost)

            

            w = w - learn_rate*gradients["dw"]# update weight

            b = b - learn_rate*gradients["db"]#update bias

        #Last parameters

        self.w = w

        self.b = b

        self.cost = self.cost_list[-1]

    def predict(self,x_test):

        x_test = x_test.T

        z = np.dot(self.w,x_test)+self.b

        y_head = 1/(1+np.exp(-z))

        for i in range(len(y_head)):

            if y_head[i] <= 0.5:

                y_head[i] = 0

            else:

                y_head[i] = 1

        return y_head

    def score(self,x_test,y_test):

        y_head = self.predict(x_test)

        return 1-np.mean(np.abs(y_head-y_test))



    

w = np.full((x_train.shape[1]),0.01)#initial weight values

b = 0.0 #initial bias value



from sklearn.preprocessing import minmax_scale

x_train = minmax_scale(x_train)#Normalize data

x_test = minmax_scale(x_test)# (x-min(x))/(max(x)-min(x))



log_reg = LogReg()

log_reg.fit(w,b,x_train,y_train,2,300)

predict = log_reg.score(x_test,y_test)

predict2 = log_reg.score(x_train,y_train)

print("Test Accurary : {}".format(predict))

print("Train Accurary : {}".format(predict2))
plt.figure(figsize = (10,7))

plt.plot(np.arange(len(log_reg.cost_list)),np.array(log_reg.cost_list))#Change of cost function

plt.title("Change of Cost Function")

plt.ylabel("Value of Cost Function")

plt.xlabel("Number of Iteration")

plt.show()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train,y_train)

predict = lr.score(x_test,y_test)

predict2 = lr.score(x_train,y_train)

print("Test Accurary : {}".format(predict))

print("Train Accurary : {}".format(predict2))