# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Reading CSV Data
data=pd.read_csv("../input/winequality-red.csv")

#Get info about data
data.info()
# Get top 10 records of data
data.head(10)
# Correlations between features
# Plot Seaborn Heatmap
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),cmap="BuPu", annot=True, linewidths=0.5)
# Boxplots of features
fig, axs = plt.subplots(2, 4, figsize = (20,10)) 
ax1 = plt.subplot2grid((5,15), (0,0), rowspan=2, colspan=3) 
ax2 = plt.subplot2grid((5,15), (0,4), rowspan=2, colspan=3)
ax3 = plt.subplot2grid((5,15), (0,8), rowspan=2, colspan=3)
ax4 = plt.subplot2grid((5,15), (0,12), rowspan=2, colspan=3)

ax5 = plt.subplot2grid((5,15), (3,0), rowspan=2, colspan=3) 
ax6 = plt.subplot2grid((5,15), (3,4), rowspan=2, colspan=3)
ax7 = plt.subplot2grid((5,15), (3,8), rowspan=2, colspan=3)
ax8 = plt.subplot2grid((5,15), (3,12), rowspan=3, colspan=3)

sns.boxplot(x='quality',y='volatile acidity', data = data, ax=ax1)
sns.boxplot(x='quality',y='citric acid', data = data, ax=ax2)
sns.boxplot(x='quality',y='sulphates', data = data, ax=ax3)
sns.boxplot(x='quality',y='alcohol', data = data, ax=ax4)

sns.boxplot(x='quality',y='fixed acidity', data = data, ax=ax5)
sns.boxplot(x='quality',y='chlorides', data = data, ax=ax6)
sns.boxplot(x='quality',y='total sulfur dioxide', data = data, ax=ax7)
sns.boxplot(x='quality',y='density', data = data, ax=ax8)
# If we analyse the data quality values range from  3 to 8 . 
sns.countplot(x='quality', data=data)
# But i need a binary result like 0:low/normal quality 1:high quality
# Most of the quality values are 5 and 6 . 
# So let's say values higher than 6 is high quality.
data['quality'] = pd.cut(data['quality'], bins = [1,6,10], labels = [0,1]).astype(int)
data.head(10)
# Let's see how many wines are high quality
sns.countplot(x='quality', data=data)
plt.show()
# Setting x and y before training data
# y=quality values [0 0 1 ... 0]
# x=values of the features, so we need to drop quality column
y=data.quality.values
x=data.drop(["quality"],axis=1)

# Normalisation
# All the values of the features will be in range 0-1
x=((x-np.min(x)) / (np.max(x)-np.min(x)))
print(x)
# train - test split data
# split data , 80% of data to train and create model
# 20% data to test model
# Use sklearn library to prepare train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Function for initializing weights and bias
# At first all the weights are 0.01 and bias is 0
def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b

# A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve
# Sigmoid functions have domain of all real numbers, with return value monotonically increasing most often from 0 to 1 or alternatively from −1 to 1, depending on convention.
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
# Forward/Backwark Propagation Functions
# z=x1w1+x2w2+...+XnWn + b
def forward_backwark_propagation(w,b,x_train,y_train):
    # forward propagation
    # calculating z
    z=np.dot(w.T,x_train) + b 
    y_head=sigmoid(z)
    # calculating loss and cost
    loss= -y_train*np.log(y_head) - (1-y_train) * np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]
    
    #backward propagation
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]
    gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost,gradients
# Function for updating parameters (weights and bias)
def update(w,b,x_traing,y_train,learning_rate,number_of_iteration):
    cost_list=[]
    cost_list2=[]
    index=[]
    
    #updating parameters number_of_iteration times
    for i in range(number_of_iteration):
        cost,gradients=forward_backwark_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        #update
        #learning rate: how fast our model will learn
        # if learning rate is too low, learning process will be slow
        # if learning rate is too high, learning process misses optimum parameters
        w = w- (learning_rate * gradients["derivative_weight"])
        b = b- (learning_rate * gradients["derivative_bias"])
        
        # add cost to cost_list2 every 10 steps
        # plot the graph by using cost_list2
        if i % 10 ==0:
            cost_list2.append(cost)
            index.append(i)
            
    #plot a graph to see the process
    parameters={"weight":w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation="vertical")
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost")
    plt.show()
        
    return parameters,gradients,cost_list
# Prediction
def predict(w,b,x_test):
    #calculate z
    z=sigmoid(np.dot(w.T,x_test) + b)    
    #fill y_prediction with zeros
    y_prediction=np.zeros((1,x_test.shape[1]))
    # if z bigger than 0.5, y_head=1 else 0
    for i in range(z.shape[1]):
        if z[0,i]<=0.5:
            y_prediction[0,i]=0
        else:
            y_prediction[0,i]=1
    return y_prediction
# Main Function
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):
    dimension=x_train.shape[0]
    # get initial parameters
    w,b=initialize_weights_and_bias(dimension)
    
    # update parameters 
    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,number_of_iteration)
    
    # predict y values according to new parameters
    y_prediction_test=predict(parameters["weight"],parameters["bias"],x_test)
    
    #calculate accuracy
    print("test accuracy {} %".format(100-np.mean(np.abs(y_prediction_test-y_test))*100))
# Running the main function
# Set learning_rate to 1 and number_of_iteration to 30
# Transpose x_train and x_test values
x_train=x_train.T
x_test=x_test.T
logistic_regression(x_train,y_train,x_test,y_test,1,30)
# set number_of_iteration to 300
logistic_regression(x_train,y_train,x_test,y_test,1,300)
#test accuracy 85.9375 %
#set number_of_iteration to 5000
logistic_regression(x_train,y_train,x_test,y_test,1,5000)
#test accuracy 86.5625 %
# Prepare train and test data (split 20% of data for testing)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# Use LogisticRegression function of sklearn
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(lr.predict(x_test))
print("test accuracy {} %".format(lr.score(x_test,y_test)*100))