# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import warnings

# filter warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from IPython.display import Image

import os

!ls ../input/



Image("../input/kaancanimg/kaan.jpg")
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp.png")
#reading data from csv file,we have two types of voice -male- or -female- ( 0 or 1 we'll say)

data=pd.read_csv("../input/voicedata/voice.csv")



#First we'll observe our features with info() command

#drop column that has no effect our classification,but our data is already clean

#check out that your data which is used to classify features cannot be object it must be

#categorical or integer !!

print("Data before arrangement: ",data.info())



#we'll convert the label column which has objects in the type of object into the integer,mean 0 or 1

#the label column must be in the type of categorical or int



data.label=[ 1 if each=='female' else 0 for each in data.label ] 

print("Data after arrangement: ",data.info()) # so now we've fixed it



#1 and 0's binary data will be in yaxis ,don't forget to convert them into numpy array

y=data.label.values

#dropping label column and assigning other columns to x_data

x_data=data.drop(['label'],axis=1)

y
x_data.tail()
data.corr()
import seaborn as sns 



f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#import the train and test library

from sklearn.model_selection import train_test_split



#we are normalizing this data,means giving every data a new value within 0 and 1

x=( x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values



#initializing the train and test datas

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

#we had the feature names as columns previously so to see it better implement them as

#their tranpose

print('x_train: ',x_train.shape )

print('Images: ',x_train.shape[0],'Pixels: ',x_train.shape[1])

print('y_train labels: ',y_train.shape )

print('x_test: ',x_test.shape )

print('Images: ',x_test.shape[0],'Pixels: ',x_test.shape[1])

print('y_test labels: ',y_test.shape )



x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T



#we've taken transposes of datas because we wanted to make it as seems in picture above

#mean taking the pixels at the left of that picture

#let's investigate the sizes of train and test data



print('\nAfter Transposes------------\n\n','x_train: ',x_train.shape )

print('Images: ',x_train.shape[0],'Pixels: ',x_train.shape[1])

print('y_train labels: ',y_train.shape )

print('x_test: ',x_test.shape )

print('Images: ',x_test.shape[0],'Pixels: ',x_test.shape[1])

print('y_test labels: ',y_test.shape )
x_train.tail()
x_test.tail()
y
y_test[:10]
def sigmoid(z):

    y_head=1/(1+np.exp(-z))

    return y_head
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp7.png")

def initialize_weights_and_bias( dimension ):

    w=np.full((dimension,1),0.01)

    b=0.0

    return w,b
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp3.png")
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp2.png")
def forward_backward_propagation(w,b,x_train,y_train):

    #forward propagation

    #weight has (30,1) dimensions,so initially create weight as (pixel_num,1)-->

    #x_train has (30,455) dimensions

    #so get the transpose of the weight matrix make it (30,1) in dimensions

    z=np.dot(w.T,x_train)+b # and make matrix multiplication (1,30)x(30,455) we get the z as ( 1,455 )

    print('z is:',z)

    y_head=sigmoid(z)

    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost=(np.sum(loss))/x_train.shape[1]

    

    #back propagation

    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias=np.sum( y_head-y_train)/x_train.shape[1]

    gradients={'derivative_weight':derivative_weight,'derivative_bias':derivative_bias}

    

    return cost,gradients

    
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp4.png")
#updating learning parameters

def update( w, b, x_train, y_train, learning_rate, number_of_iteration ):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    #updating learning parameters is number of iteration times

    

    for i in range( number_of_iteration):

        #make forward and backward propagation and find cost ad gradients

        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        #update process

        w=w-learning_rate*gradients["derivative_weight"]

        b=b-learning_rate*gradients["derivative_bias"]

        

        if i % 10 == 0:

           cost_list2.append(cost)

           index.append(i)

           #print('Cost after iteration %i: %f' %(i,cost))

    

   #we update(learn) parameters weight and bias

   

    parameters={'weight':w,'bias':b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel('Number Of Iteration')

    plt.ylabel('Cost')

    plt.show()

    return parameters,gradients,cost_list

from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp5.png")
 # prediction

def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b) #probabilistic result

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp6.png")
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp8.png")
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 1500)
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 3000)
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 5000)
from IPython.display import Image

import os

!ls ../input/



Image("../input/screen/pp9.png")
from sklearn import linear_model



logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)



print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))

print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))