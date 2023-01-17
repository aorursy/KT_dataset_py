# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visualization
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/voice.csv")
df.head()
#Just having some information about our dataset

# import data, see feature names, label count and data info
print(df.columns)
label_value_count = df.label.value_counts()
print(label_value_count)
print(df.info())
#correlation map
f,ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, linewidths=.8, fmt= '.1f',ax=ax)
sns.jointplot(df['meanfreq'], df['sfm'], kind="regg", color="red")
male = []
female = []
for i in range(0,3167):
    for j in range(0,20):
        if df.iloc[i,20] == "male":
            male.append([df.iloc[i,j]])
        elif df.iloc[i,20] == "female":
            female.append([df.iloc[i,j]])      
sum_male = 0
size_male = len(male) #1584
for i in range(0,size_male,20):
    sum_male = np.add(sum_male,male[i:i+20])

sum_female =0
size_female = len(female) #1583
for i in range(0,size_female,20):
    sum_female = np.add(sum_female,female[i:i+20])

sum_male = sum_male/1584
sum_female = sum_female/1583
data = {"Male":[sum_male],"Female":[sum_female]}
data = pd.DataFrame(data)
data
#changing "male" and "female" names by '1' and '0' 
df.label = [1 if i == "male" else 0 for i in df.label]
y = df.label.values   #we store the labels in y
x_data = df.drop(["label"], axis=1)  #we put everything except label into x_data
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data))  #normalization of the data.
#we divide our data into some percentages that we desired, like %20 for test and %80 for train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
#since we apply matrix multiplication, we take transforms of our matrices
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
#We initialize the weightt and bias values by default
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1), 0.01) #'w' is a vector which has the same dimension with our features
    b = 0.0
    
    return w,b
#we define our sigmoid function which gives us the resulting labels.
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    
    return y_head
#as we can see from the picture at the beginning, we do the forward propagation first and,
#we apply backward propagation steps to get the parameters for updating the weight and bias values.
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients
#we use the resulting values in the previous funtion to update our 'w' and 'b'
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
#Here is the prediction part, we determine whether the result should be 1 or 0 according to our 'w' and 'bias' values
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
#This is the final part for logistic regression. 
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1.5, num_iterations = 501)   
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(x_train.T,y_train.T)
print("test accuracy {}".format(logistic_regression.score(x_test.T,y_test.T)))