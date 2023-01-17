# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from matplotlib import pyplot as plt

import random

#sources:https://towardsdatascience.com/gradient-descent-in-python-a0d07285742f

#
#read the dataset

iris=pd.read_csv("../input/Iris.csv")
#get the first 5 elements

iris.head()
#get length of the iris dataset

print(len(iris.SepalLengthCm))
m,n=iris.shape

#get the first 100 values as training set

x_train=iris.SepalLengthCm.iloc[0:100].values.reshape(100,1)

y_train=iris.SepalWidthCm.iloc[0:100].values.reshape(100,1)

#get last 50 elemennt as test set

x_test=iris.SepalLengthCm.iloc[101:149].values.reshape(48,1)

y_test=iris.SepalWidthCm.iloc[101:149].values.reshape(48,1)

#visualitzation of data

plt.scatter(x_train,y_train)

plt.xlabel("Sepal length")

plt.ylabel("sepal width")

plt.show()
#assign random number for weight

weight=np.random.randn(1)

bias=np.random.randn(1)

#length of datapoints

N=len(x)



iterations=2000 #number of iteration

learning_rate=0.001



def gradient_descent(weight_,bias_,N_,learning_rate_,iteration,y_,x_):

    past_costs=[]

    past_weight=[]

    

    for i in range(iteration):

        #y_pred=mx+b where m is weight and b is bias

        y_pred=(x_*weight_)+bias_

        

        #transpose the x_ matrix . matrix is now [row=1:column=100]

        x_trans=x_.T

        

        #print((np.dot(x_trans,(y_pred-y_))))

        #gradient descent formula

        #(1/number of samples) (summation of (x*(error)) -->done using dot matrix)

        #where error is y_pred-y_train 

        #dot is [1:100]dot[100:1] resulting matrix is [1:1]

        weight_=weight_-(((1/N_)*learning_rate_)*((np.dot(x_trans,(y_pred-y_)))))

        #print((((1/N_)*learning_rate_)*(((np.dot(x_trans,(y_pred-y_)))))))

     

        #cost function

        #sum of square error

        loss_calc=(np.sum((y_pred-y_)**2)/(2*N_))

        

        #get the past costs and past weights for plotting 

        #past_costs.append(loss_calc)

        #past_weight.append(weight_)

        

    

    #weight_calc=weight_

    #print(weight_calc)

    #loss_calc=loss_calc

    return weight_,loss_calc

        



weight_ans,loss_ans=gradient_descent(weight,bias,N,learning_rate,iterations,y_train,x_train)
#the value of weight bbased on our training

print(weight_ans)
#get the ypre_test using x_test data, trained weight and bias

ypred_test= (x_test*weight_ans)+bias

print(ypred_test)
#get the accuracy using Mean Absolute Percent Error

#MAPE= summation of(|actual- predicted|/actual) *100/Num_samples or

accuracy=np.mean(np.abs((y_test - ypred_test)) / y_test) * 100

print(accuracy)



#model still not accurate with large error