# LİBRARY ADD 

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))

# We are reading our data

data = pd.read_csv('../input/cardio_train.csv' , sep = ';' )

data.head()

# id  unnecessary 

# id is drop 

data.drop(['id'] ,axis = 1 ,inplace = True)



y = data.cardio.values

# CARDİO out of Data 

x_data = data.drop(['cardio'],axis = 1 )

x_data.head()
# normalization 

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

x.head()



from sklearn.model_selection import train_test_split 

x_train , x_test ,y_train , y_test =train_test_split(x , y  , test_size = 0.2 , random_state = 42 )



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T





print("x_train: ",x_train.shape)

print("x_test: ",x_test.shape)

print("y_train: ",y_train.shape)

print("y_test: ",y_test.shape)
# weigh and bias function 

#parameter initialize and sigmoid function 

#dimension  = 11 :) 

def initialize_weights_and_bias(dimension):

    w = np.full((dimension , 1 ) , 0.01)

    b = 0.0

    return w,b 



# example 

#print(initialize_weight_and_bias(11))



def sigmoid(z):

    y_head = 1 / (1 + np.exp(-z))

    return y_head 







def forward_backward_propagation(w,b,x_train,y_train):

    #forward propagation

    z = np.dot(w.T ,x_train) + b 

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/ x_train.shape[1]

    

    #backward propagations

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients

    



    
#updating 

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
#predict 

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

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 100)    

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)  # n_neighbors means k

knn.fit(x_train.T, y_train.T)

prediction = knn.predict(x_test.T)



print("{} NN Score: {:.2f}%".format(3, knn.score(x_test.T, y_test.T)*100))

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.T, y_train.T)

print("Accuracy of Naive Bayes: {:.2f}%".format(nb.score(x_test.T,y_test.T)*100))