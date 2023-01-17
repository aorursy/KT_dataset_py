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
df=pd.read_csv("../input/voice.csv") # import csv file
df.label=[1 if each=="male" else 0 for each in df.label] # make values 0 or 1


y=df.label.values

xData=df.drop(["label"], axis=1)





x=(xData-np.min(xData))/(np.max(xData)-np.min(xData))



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T















def initialize_weights_and_bias(dimension): #initial values of weight and bias

    w= np.full((dimension,1),0.01)

    b= 0.0

    return w,b
def sigmoid(z): ## sigmoid function

    y_head=1/(1+np.exp(-z))

    return y_head
def forward_backward_propogation(w,b,x_train,y_train): ## forward and backward propogation

    z=np.dot(w.T,x_train)+b

    y_head=sigmoid(z)



    loss=-(y_train*np.log(y_head)+(1-y_train)*np.log(1-y_head))

    cost=(np.sum(loss))/x_train.shape[1]



    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]



    gradients={"derivative_weight" : derivative_weight,

          "derivative_bias" : derivative_bias}



    return cost,gradients



    

def update(w,b,x_train,y_train,learning_rate,number_of_iteration): ### new values for weight and bias

    index=[]

    cost_list=[]

    cost_list2=[]

    

    for i in range(number_of_iteration):

        

        cost,gradients=forward_backward_propogation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w=w-learning_rate * gradients["derivative_weight"]

        b=b-learning_rate * gradients["derivative_bias"]

        

        if i%10==0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i : %f" %(i,cost))

    

    parameters={"weight": w , "bias" : b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation="vertical")

    plt.xlabel("Iteration")

    plt.ylabel("cost")

    plt.title("Cost graph")

    plt.show()

    

    return parameters,gradients,cost_list

        
def predict(w,b,x_test):  ### prediction

    

    z=sigmoid(np.dot(w.T,x_test)+b)

    y_prediction=np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<=0.5:

            y_prediction[0,i]=0

        else:

            y_prediction[0,i]=1

            

    return y_prediction

        

    

    
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration): ## results

    

    dimension=x_train.shape[0]

    w,b=initialize_weights_and_bias(dimension)

    parameters,gradients,cost_list=update(w,b,x_train,y_train,learning_rate,number_of_iteration)

    y_prediction_test=predict(parameters["weight"],

                              parameters["bias"],x_test)

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(x_train,y_train,x_test,y_test,learning_rate=1,number_of_iteration=50)


