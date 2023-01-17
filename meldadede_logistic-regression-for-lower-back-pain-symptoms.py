#Import necessary libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #for visiulazitions

#Load data set
data = pd.read_csv("../input/Dataset_spine.csv")
data.info() #gives information about column types and NaN values
data.head(10) #gets first 10 row of the data
#Store attribute names which are written in "Unnamed: 13" column, then delete the column
attribute_names = data.iloc[5:17]["Unnamed: 13"]
data.drop(["Unnamed: 13"],axis=1,inplace=True)
#Convert two class names to 0 and 1
data.Class_att = [1 if each == "Abnormal" else 0 for each in data.Class_att]
data.describe() #gives some statistical values about data set, such as max,min,mean,median etc.
#assign Class_att column as y attribute
y = data.Class_att.values

#drop Class_att column, remain only numerical columns
new_data = data.drop(["Class_att"],axis=1)

#Normalize values to fit between 0 and 1. 
x = (new_data-np.min(new_data))/(np.max(new_data)-np.min(new_data)).values
#Split data into Train and Test 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state =42)
#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
#parameter initialize and sigmoid function
def initialize_weights_and_bias(dimension):
    
    w = np.full((dimension,1),0.01) #first initialize w values to 0.01
    b = 0.0 #first initialize bias value to 0.0
    return w,b
#sigmoid function fits the z value between 0 and 1
def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    
    y_head = sigmoid(np.dot(w.T,x_train) + b)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # backward propagation
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}
    
    return cost,gradients
def update_weight_and_bias(w,b,x_train,y_train,learning_rate,iteration_num) :
    cost_list = []
    index = []
    
    #for each iteration, update w and b values
    for i in range(iteration_num):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
        cost_list.append(cost)
        index.append(i)

    parameters = {"weight": w,"bias": b}
    
    print("iteration_num:",iteration_num)
    print("cost:",cost)

    #plot cost versus iteration graph to see how the cost changes over number of iterations
    plt.plot(index,cost_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients
def predict(w,b,x_test):
    z = np.dot(w.T,x_test) + b
    y_predicted_head = sigmoid(z)
    
    #create new array with the same size of x_test and fill with 0's.
    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_predicted_head.shape[1]):
        if y_predicted_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,iteration_num):
    dimension = x_train.shape[0]#For our dataset, dimension is 248
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients = update_weight_and_bias(w,b,x_train,y_train,learning_rate,iteration_num)

    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
    # Print test Accuracy
    print("manuel test accuracy:",(100 - np.mean(np.abs(y_prediction - y_test))*100)/100)
logistic_regression(x_train,y_train,x_test,y_test,1,10)
logistic_regression(x_train,y_train,x_test,y_test,10,10)
logistic_regression(x_train,y_train,x_test,y_test,4,200)
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression()

lr_model.fit(x_train.T,y_train.T)

print("sklearn test accuracy:", lr_model.score(x_test.T,y_test.T))
