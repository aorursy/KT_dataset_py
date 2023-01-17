        

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



data= pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

data.drop(["Unnamed: 32","id"],axis=1,inplace=True)



data.diagnosis=[1 if each =="M" else 0 for each in data.diagnosis]

print(data.info())



y = data.diagnosis.values # Converting Diagnosis to numpy

x_data = data.drop(["diagnosis"],axis=1) #Removing All Diagnosis Data from x 



#%% Normalization

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)) # Implementation of Normalization

#%% Dataset Train-Test Split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42) # Splitting our dataset into %80-%20 ratio

x_train=x_train.T

x_test=x_test.T

y_train=y_train.T

y_test=y_test.T



#%% Implementation of Initializing Parameters and Sigmoid



def initialize_w_b(dimension):

    w=np.full((dimension,1),0.01) # Creating a vector with 30 rows and full of 0.01

    b=0.0

    return w,b



def sigmoid(z):

    y_head=1/(1+np.exp(-z)) #Sigmoid Function

    return y_head

#%% Implementation of Forward and Backward Propagation

import numpy as np

def forw_back_propagation(w,b,x_train,y_train):

    #Forward Part

    z=np.dot(w.T,x_train)+b  #Multiplication of all weights with our x_train values  

    y_head=sigmoid(z) #Converting Z into a number which will take value between 0-1

    loss=-(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head) #Calculation of Loss Function by Using Log Loss Function

    cost=(np.sum(loss))/(x_train.shape[1]) #Each loss summed and divided into how many data we have

    

    #Backward Part

    deri_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] #Derivation of Weight for using later to update and optimize

    deri_bias=np.sum(y_head-y_train)/x_train.shape[1] #Derivation of Bias for using later to update and optimize

    gradients={"deri_weight":deri_weight,"deri_bias":deri_bias} #Creation of Gradients Dictionary

    

    return cost,gradients



#%% Implementation of Update of Parameters

def update(w,b,x_train,y_train,learning_rate,n_iterations):

    cost_list=[]

    cost_list2=[]

    index=[]

    

    for i in range(n_iterations): #Calling Cost and Gradient to update in below with formulas starts with w & b 

        cost,gradients = forw_back_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w=w-learning_rate*gradients["deri_weight"]

        b=b-learning_rate*gradients["deri_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

            

    parameters = {"weight": w,"bias":b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list  

#%% Prediction     

    

def predict(w,b,x_test): 

    

    z= sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            Y_prediction[0,i]=0

        else:

            Y_prediction[0,i]=1

    

    return Y_prediction

#%% Logistic Regression



def LogisticReg(x_train,y_train,x_test,y_test,learning_rate,n_iterations):

    dimension = x_train.shape[0]

    w,b = initialize_w_b(dimension)

    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,n_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test) #Prediction of Y with parameters Weight and Bias

    print("Test Accuracy:{} %".format(100 -np.mean(np.abs(y_prediction_test-y_test))*100)) #Printing Out Test Accuracy

    

LogisticReg(x_train,y_train,x_test,y_test,learning_rate=1,n_iterations=300) #An example of usage with rate =1 and iterations =300
