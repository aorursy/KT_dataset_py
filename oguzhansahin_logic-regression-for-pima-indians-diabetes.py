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
#Importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#Importing our data



data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")



print("Number of features : ",data.shape[1])

print("Number of samples  :" ,data.shape[0])
data.info()
data.columns
data.head(5)
data.shape
y = data['Outcome'].values



x_data = data.drop(['Outcome'],axis=1)





#Visualization

f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot = True , linewidth = .5, fmt = '.2f',ax = ax)

plt.show()
data.plot(kind="scatter",x="Glucose",y="Outcome",color="blue")

plt.xlabel("Glucose")

plt.ylabel("Outcome")

plt.show()
#Normalization



x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values

x
#train-test split



from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
#Transpoze



x_train = x_train.T

x_test  = x_test.T

y_train = y_train.T

y_test  = y_test.T
#initializing parameters and sigmoid function



def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b=0.0

    return w,b



#%%



def sigmoid(z):

    

    y_head = 1/(1+np.exp(-z))

    return y_head



#Forward and Backward Propogation



def forward_backward_propagation(w,b,x_train,y_train):

    

    #forward propogation

    

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

     

    # backward propogation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients

    
#Updating parameters

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    

    cost_list  = []

    cost_list2 = []

    index      = []

    

    # updating parameters is number_of_iterarion times

    

    for i in range(number_of_iterarion):

        

        # make forward and backward propagation and find cost and gradients

       

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        # lets update

        

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        

        if i % 30 == 0:

            

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
#Prediction

def predict(w,b,x_test):

    

    # x_test is a input for forward propagation

    

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    # if z is bigger than 0.5, our prediction is one (y_head=1),

    # if z is smaller than 0.5, our prediction is zero (y_head=0),

    

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

    

logistic_regression(x_train,y_train,x_test,y_test,1,700)
# sklearn with LR





from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)



print("test accuracy {}".format(lr.score(x_test.T,y_test.T))) 