import numpy as np # Linear algebra

import pandas as pd # Data processing.

import matplotlib.pyplot as plt # Visualize cost function
df = pd.read_csv("../input/data.csv") # Import data

print(df.columns)

df.drop(["id","Unnamed: 32"],axis = 1,inplace = True) # We delete Id and Unnamed 32 because they have nothing to do with machine learning
df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis] # We convert B and M to 1 and 0 because our data must be binary numeric

y = df.diagnosis.values # We take diagnosis values

x_data = df.drop(["diagnosis"], axis = 1) # We take all features outside diagnosis

x = (x_data -np.min(x_data))/(np.max(x_data)-np.min(x_data)).values # Normalization
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # 80% train, 20% test



print(x_train.shape)

x_train = x_train.T # Transpose

x_test = x_test.T # Transpose

print(x_train.shape)

# We didn't transpose y_train and y_test because they are vectorial
def initalize_weight_bias(dimension):

    # dimension -> Number of Feature

    w = np.full((dimension,1),0.1)

    b = 0.0

    return w,b
def sigmoid(z):

    y_head = 1/(1+np.exp(-z)) # y_head -> predicted value

    return y_head
def forward_backward_propagation(w,b,x_train,y_train):

    # Forward Propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z) # y_head -> predicted value

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1] # Our purpose is minimizing cost

    

    # Backward Propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # derivative of weight

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1] # derivative of bias

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
def update(w, b, x_train, y_train, learning_rate, num_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    for i in range(num_of_iteration):

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        w = w - learning_rate * gradients["derivative_weight"] # Update weight against to learning_rate and derivative

        b = b - learning_rate * gradients["derivative_bias"] # Update bias against to learning_rate and derivative

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

        

    parameters = {"weight": w,"bias": b}

        

    # Visualize cost_list

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w,b, x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_prediction = np.zeros((1,x_test.shape[1])) # We've created an array equal to the number of feature

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            y_prediction[0,i] = 0

        else:

            y_prediction[0,i] = 1

    return y_prediction
def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,num_of_iterations):

    dimension = x_train.shape[0] # Number of feature

    w,b = initalize_weight_bias(dimension)

    parameters, gradients, cost_list = update(w,b,x_train, y_train, learning_rate, num_of_iteration)

    

    test_prediction = predict(parameters["weight"],parameters["bias"], x_test) # We predict against to x_test

    train_prediction = predict(parameters["weight"],parameters["bias"], x_train) # We predict against to x_train

    

    print("train accuracy: {} %".format(100 - np.mean(np.abs(train_prediction - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(test_prediction - y_test)) * 100))

    print("cost: {}".format(min(cost_list)))

    return test_prediction
learning_rate = 5

num_of_iteration = 300

test_prediction = logistic_regression(x_train,x_test,y_train,y_test,learning_rate,num_of_iteration).tolist()[0]
import seaborn as sns

import matplotlib.pyplot as plt



y_predict = test_prediction # test_prediction -> predicted values

y_true = y_test



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_predict) # We use confusion matrix for comparison



f, ax = plt.subplots(figsize = (5,5))

sns.heatmap(cm,annot = True, linewidths = 0.5, linecolor = "red", fmt = ".0f",ax = ax)

plt.xlabel("predicted")

plt.ylabel("real")

plt.show()

# As you can see, our model has estimated 3 values wrong