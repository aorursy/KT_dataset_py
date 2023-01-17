# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Create a simple dataset from dictionary.
dictionary_1 = {"dogru_soru_sayisi":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              "puan":[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]}
data_1 = pd.DataFrame(dictionary_1)
data_1.head(20)
#Visualize the data
plt.scatter(data_1.dogru_soru_sayisi,data_1.puan)
plt.xlabel("dogru_soru_sayisi")
plt.ylabel("puan")
plt.show()
#Lets look a type
type(data_1.dogru_soru_sayisi)
#Want a array
x = data_1.dogru_soru_sayisi.values
y = data_1.puan.values
type(x)
#How we x of shape?
x.shape
#Want shape (20,1)
x = data_1.dogru_soru_sayisi.values.reshape(-1,1)
y = data_1.puan.values.reshape(-1,1)
x.shape
#We need sklearn library
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
#We should make fit
lr.fit(x,y)
#Visualize
plt.scatter(x,y)
plt.xlabel("dogru_soru_sayisi")
plt.ylabel("puan")
#We should make predict
y_head = lr.predict(x)
#Visualize
plt.plot(x,y_head, color = "red")
plt.show()
#Create a simple dataset from dictionary.
dictionary_2 = {"enerji":[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                "guc":[10,22,34,46,56,61,70,83,97,100,101,104,109,109,118,120,123,123,123,124]}
data_2 = pd.DataFrame(dictionary_2)
data_2.head(20)
#How we make polynomial regression on python?
x2 = data_2.enerji.values.reshape(-1,1)
y2 = data_2.guc.values.reshape(-1,1)
#Library
from sklearn.linear_model import LinearRegression
lr2 = LinearRegression() 
#Fit
lr2.fit(x2,y2)
#Visualize
plt.scatter(x2,y2, color = "blue") #our values
plt.ylabel("guc")
plt.xlabel("enerji")
#Predict
y_head2 = lr.predict(x2)
#Visualize
plt.plot(x2, y_head2 , color="red" )  #that does not represent our values well

#Library
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4)  #degree of the polynomial
#Fit
x_polynomial = polynomial_regression.fit_transform(x2)
lr3 = LinearRegression()
lr3.fit(x_polynomial,y2)
#Predict
y_head3 = lr3.predict(x_polynomial)
#Visualize
plt.plot(x2,y_head3, color = "green")  #this is better
plt.show()

#How we use decision tree regression on python?
#We used the data(data_1) we created above.
x = data_1.dogru_soru_sayisi.values.reshape(-1,1)
y = data_1.puan.values.reshape(-1,1)
#Library
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
#Fit
dtr.fit(x,y)
#Step
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
#Predict
y_head4 = dtr.predict(x_)
#Visualize
plt.scatter(x,y,color = "red")
plt.plot(x_,y_head4, color = "green")
plt.show()


#We build data frames from csv.
df = pd.read_csv("../input/voice.csv")
df.head()
df.info()
#df = pd.read_csv("../input/voice.csv")
df.label.unique()
df.label = [ 1 if each == "male" else 0 for each in df.label]
df.label.unique()
y = df.label.values
x_df = df.drop(["label"], axis = 1)
#Normalization on python
x = (x_df - np.min(x_df))/(np.max(x_df) - np.min(x_df)).values

#Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train:",x_train.shape)
print("x_test:",x_test.shape)
print("y_train:",y_train.shape)
print("y_test:",y_test.shape)
#Parameter initialize 
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w , b
#Sigmoid function
def sigmoid(z):
    y_head_ = 1/(1 + np.exp(-z))
    return y_head_

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head_ = sigmoid(z)
    loss = -y_train*np.log(y_head_)-(1-y_train)*np.log(1-y_head_)
    cost = (np.sum(loss))/x_train.shape[1]
    
    #backward propagation
    derivative_weight = (np.dot(x_train,((y_head_-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head_-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost,gradients
#Updating parameters
def update(w,b,x_train,y_train, learning_rate, num_iterations):
    cost_list = []
    cost_list2 = []
    index = []
    
    #Updating parameters is num_iteration times
    for i in range(num_iterations):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("cost after iteration %i: %f" %(i,cost))
            
            
    #We update parameters weights and bias
    parameters = {"weight": w, "bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters,gradients,cost_list
#Prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_predict = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head_=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head_=0)
    for i in range (z.shape[1]):
        if z[0,i] <= 0.5:
            y_predict[0,i] = 0
        else:
            y_predict[0,i] = 1
            
    return y_predict
#Logistic regression
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):
    # initialize
    dimension = x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w,b,x_train,y_train, learning_rate, num_iterations)
    
    y_predict_test = predict(parameters["weight"],parameters["bias"],x_test)
    
    # print train/test errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict_test - y_test))*100))
    
logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, num_iterations= 300)
#Library
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train.T,y_train.T)
print("test accuracy {}".format(logistic_regression.score(x_test.T,y_test.T)))
