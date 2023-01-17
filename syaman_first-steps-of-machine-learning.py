import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#first we must read our data to use it 
data=pd.read_csv("../input/first-data/ab.csv",sep=";")

#ploting our data
plt.scatter(data.speed,data.time)
plt.xlabel("speed")
plt.ylabel("time")
plt.title("speed & time")
plt.show()
#define linear regression
linear_reg=LinearRegression()
#with ".values" x will be numpy type and to make shape (14,1) we must use ".reshape(-1,1)"
#if we don't use .reshape, we will see our data like (14,)

x=data.time.values.reshape(-1,1)
y=data.speed.values.reshape(-1,1)

plt.scatter(x,y)


linear_reg.fit(x,y) # with this commmand we will draw the fit line of linear regression

array=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)
y_head=linear_reg.predict(array)

plt.plot(array,y_head,color="red")
plt.show()


x=data.iloc[:,[0,2]].values
y=data.speed.values.reshape(-1,1)
multiple_LN=LinearRegression()
multiple_LN.fit(x,y)
multiple_LN.predict(np.array([[10,35],[5,35]]))

#we will see from the graph that will show below, linear regression is not suitable for any data

from sklearn.linear_model import LinearRegression
pr=pd.read_csv("../input/poly-reg/poly_Regr.csv",sep=";")

x=pr.CarSpeed.values.reshape(-1,1)
y=pr.CarCost.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("Speed-car")
plt.ylabel("Price-car")

LR=LinearRegression()
LR.fit(x,y)
y_head=LR.predict(x)
plt.plot(x,y_head,color="red",Label="linear")

#import polinomial regression
from sklearn.preprocessing import PolynomialFeatures

pol_reg=PolynomialFeatures(degree=4)
#with degree we can make change degree of polinom degree, mean that x^4

x_poly=pol_reg.fit_transform(x)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
y_head2=lin_reg2.predict(x_poly)

plt.scatter(x,y)
plt.xlabel("Speed-Car")
plt.ylabel("Cost-Car")

plt.plot(x,y_head2,color="green",label="poly")
plt.legend()
plt.show()
#we will see 2 dimentions of decision tree graph with the below codes
from sklearn.tree import DecisionTreeRegressor

Dt=pd.read_csv("../input/decs-tree/desicionTree.csv",sep=";")

x=Dt.one.values.reshape(-1,1)
y=Dt.two.values.reshape(-1,1)

dec_tree=DecisionTreeRegressor()
dec_tree.fit(x,y)

x_=np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head=dec_tree.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribun")
plt.ylabel("Cost")
plt.show()
from sklearn.ensemble import RandomForestRegressor

rfr=pd.read_csv("../input/decs-tree/desicionTree.csv",sep=";")

x=rfr.one.values.reshape(-1,1) # first column of our data
y=rfr.two.values.reshape(-1,1) # second column of our data

rf=RandomForestRegressor(n_estimators=100,random_state=21) 
#n_estimators= number of trees

rf.fit(x,y)
x_= np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head=rf.predict(x_)

plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="blue")
plt.xlabel("Tribun-Level")
plt.ylabel("Cost")
plt.show()


#we will use first_data
from sklearn.metrics import r2_score
data=pd.read_csv("../input/first-data/ab.csv",sep=";")

x=data.speed.values.reshape(-1,1)
y=data.time.values.reshape(-1,1)

linear_reg.fit(x,y)
y_head=linear_reg.predict(x)

r2_score(y,y_head)
#we will seee our result that is 0.92899.... , this answer is near to 1, 
#because of this, we can say that our regression model is good
#first we will read our data with pandas library
ex=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

#ploting scatter graph
x=ex.lumbar_lordosis_angle.values.reshape(-1,1)
y=ex.sacral_slope.values.reshape(-1,1)

plt.scatter(x,y,color="green")
plt.xlabel("lumbar_lordosis_angle")
plt.ylabel("sacral_slope")

#ploting linear regression line
LR=LinearRegression()
LR.fit(x,y)
y_head=LR.predict(x)
plt.plot(x,y_head,color="red",Label="linear")
plt.show()

#last step, we can evalate our linear regression data with r-square method

r2_score(y,y_head)

#we will see our result that is 0.35806... and this result is not good, our linear regression model is not good for this data set
#we can make polinomial regression model, will be better for our data 
from sklearn.preprocessing import PolynomialFeatures
pol_reg=PolynomialFeatures(degree=4)

ex=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

#ploting scatter graph
x=ex.lumbar_lordosis_angle.values.reshape(-1,1)
y=ex.sacral_slope.values.reshape(-1,1)

x_poly=pol_reg.fit_transform(x)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)
y_head2=lin_reg2.predict(x_poly)

plt.scatter(x,y,color="green")
plt.xlabel("lumbar_lordosis_angle")
plt.ylabel("sacral_slope")

plt.plot(x,y_head2,color="red",label="poly")
plt.legend()
plt.show()

# We will evalate our data
r2_score(y,y_head2)
#we will see result of regression evaluation model that is 0.444661..., it is better then linear regression model
ex=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv")

#ploting scatter graph
x=ex.lumbar_lordosis_angle.values.reshape(-1,1)
y=ex.sacral_slope.values.reshape(-1,1)

#we define our model
reg=LinearRegression()

#predict space
predict_space=np.linspace(min(x),max(x)).reshape(-1,1)

#fit
reg.fit(x,y)
#predict
predict=reg.predict(predict_space)

#ploting
plt.scatter(x,y)
plt.plot(predict_space,predict,color="red",linewidth=3)
plt.xlabel("lumbar_lordosis_angle")
plt.ylabel("sacral_slope")
plt.show()

#we can evaluate our result

reg.score(x,y)
logr=pd.read_csv("../input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv")

#list comprehension
logr["class"] = [1 if each=="Normal" else 0 for each in logr["class"]]

y=logr["class"].values.reshape(-1,1)
x_data=logr.drop(["class"],axis=1)

# %% normalization
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# (x - min(x))/(max(x)-min(x))

# %% train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)

# %% parameter initialize and sigmoid function
# dimension = 30
def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

# w,b = initialize_weights_and_bias(30)

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head
# print(sigmoid(0))

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients

#%% Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list

#%%  # prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction

# %% logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 40)    

#Logistic Regression with Sklearn Library

from sklearn.linear_model import LogisticRegression

LR=LinearRegression()
LR.fit(x_train.T,y_train.T)

print("test_accuracy {}".format(LR.score(x_test.T,y_test.T)))
#finished:)