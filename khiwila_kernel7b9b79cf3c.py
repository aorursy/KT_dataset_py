# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def convert_to_one_hot(Y, C):

    Y = np.eye(C)[Y.reshape(-1)].T

    return Y


train= pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")

print(train.shape)

print(test.shape)
train_data = train.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'], 1)

test_data = test.drop(['PassengerId','Name','Ticket','Cabin', 'Embarked'], 1)

train_data['Sex'].replace(['female','male'],[0,1],inplace=True)

test_data['Sex'].replace(['female','male'],[0,1],inplace=True)

train_data['Age'].replace(np.nan,30,inplace=True)

test_data['Age'].replace(np.nan,30,inplace=True)

test_data['Pclass'].replace(np.nan,1,inplace=True)

train_data['Pclass'].replace(np.nan,1,inplace=True)

test_data['SibSp'].replace(np.nan,1,inplace=True)

train_data['SibSp'].replace(np.nan,1,inplace=True)

test_data['Fare'].replace(np.nan,10,inplace=True)

train_data['Fare'].replace(np.nan,10,inplace=True)

print(train_data.shape)

print(test_data.shape)

print(train_data.head())

print(test_data.head())
print(train_data.describe)

print(test_data.describe)
X=np.array(train_data.drop(['Survived'],1))

max_train=np.max(X, axis=0)

X=(X / max_train).T



Y=np.reshape(np.array(train_data['Survived']), (1,X.shape[1]))

Y= convert_to_one_hot(Y, 2)



X_test=np.array(test_data)

max_test=np.max(X_test, axis=0)

X_test=(X_test / max_test).T



print(X.shape)

print(Y.shape)

print(X_test.shape)

def sigmoid(x):

    """

    Compute the sigmoid of x



    Arguments:

    x -- A scalar or numpy array of any size.



    Return:

    s -- sigmoid(x)

    """

    s = 1/(1+np.exp(-x))

    return s
def layer_sizes(X, Y):

    """

    Arguments:

    X -- input dataset of shape (input size, number of examples)

    Y -- labels of shape (output size, number of examples)

    

    Returns:

    n_x -- the size of the input layer

    n_h -- the size of the hidden layer

    n_y -- the size of the output layer

    """



    n_x = X.shape[0] # size of input layer

    n_h = 100

    n_y = 2 # size of output layer



    return (n_x, n_h, n_y)
(n_x, n_h, n_y)=layer_sizes(X,Y)

print(n_x, n_h, n_y)
def initialize_parameters(n_x, n_h, n_y):

    """

    Argument:

    n_x -- size of the input layer

    n_h -- size of the hidden layer

    n_y -- size of the output layer

    

    Returns:

    params -- python dictionary containing your parameters:

                    W1 -- weight matrix of shape (n_h, n_x)

                    b1 -- bias vector of shape (n_h, 1)

                    W2 -- weight matrix of shape (n_y, n_h)

                    b2 -- bias vector of shape (n_y, 1)

    """



    W1 = np.random.randn(n_h, n_x)*0.01

    b1 = np.zeros((n_h, 1))

    W2 = np.random.randn(n_y, n_h)*0.01

    b2 = np.zeros((n_y, 1))



    assert (W1.shape == (n_h, n_x))

    assert (b1.shape == (n_h, 1))

    assert (W2.shape == (n_y, n_h))

    assert (b2.shape == (n_y, 1))

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

   

    return parameters
def forward_propagation(X, parameters):



    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]



    # Implement Forward Propagation to calculate A2 (probabilities)



    Z1 = np.dot(W1,X)+b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2,A1)+b2

    A2 = sigmoid(Z2)



    cache = {"Z1": Z1,

             "A1": A1,

             "Z2": Z2,

             "A2": A2}

 

    return A2, cache
def compute_cost(A2, Y, parameters):



    m = Y.shape[1] # number of example

    

    # Compute the cross-entropy cost



    logprobs = np.multiply(np.log(A2),Y)

    cost = - np.sum(logprobs)

    

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 

                               

    assert(isinstance(cost, float))

    

    return cost
def backward_propagation(parameters, cache, X, Y):



    m = X.shape[1]

       

    # First, retrieve W1 and W2 from the dictionary "parameters".

    W1 = parameters["W1"]

    W2 = parameters["W2"]

       

    # Retrieve also A1 and A2 from dictionary "cache".

    ### START CODE HERE ### (≈ 2 lines of code)

    A1 = cache["A1"]

    A2 = cache["A2"]

    

    # Backward propagation: calculate dW1, db1, dW2, db2. 

    dZ2 = A2-Y

    dW2 = 1/m*np.dot(dZ2,A1.T)

    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))

    dW1 = 1/m*np.dot(dZ1,X.T)

    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)

    

    grads = {"dW1": dW1,

             "db1": db1,

             "dW2": dW2,

             "db2": db2}

    

    return grads
def update_parameters(parameters, grads, learning_rate = 1.2):



    # Retrieve each parameter from the dictionary "parameters"



    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]



    # Retrieve each gradient from the dictionary "grads"



    dW1 = grads["dW1"]

    db1 = grads["db1"]

    dW2 = grads["dW2"]

    db2 = grads["db2"]

    

    # Update rule for each parameter



    W1 = W1-learning_rate*dW1

    b1 = b1-learning_rate*db1

    W2 = W2-learning_rate*dW2

    b2 = b2-learning_rate*db2



    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

  

    return parameters
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):



    n_x = layer_sizes(X, Y)[0]

    n_y = layer_sizes(X, Y)[2]

    

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".



    parameters = initialize_parameters(n_x, n_h, n_y)

    W1 = parameters["W1"]

    b1 = parameters["b1"]

    W2 = parameters["W2"]

    b2 = parameters["b2"]



    # Loop (gradient descent)



    for i in range(0, num_iterations):

        

        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".

        A2, cache = forward_propagation(X, parameters)

       

        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".

        cost = compute_cost(A2, Y, parameters)

 

        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".

        grads = backward_propagation(parameters, cache, X, Y)

 

        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".

        parameters = update_parameters(parameters, grads, learning_rate = 2.1)



        # Print the cost every 1000 iterations

        if print_cost and i % 1000 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    

    return parameters
parameters = nn_model(X, Y, 100, num_iterations=30000, print_cost=True)

#print("W1 = " + str(parameters["W1"]))

#print("b1 = " + str(parameters["b1"]))

#print("W2 = " + str(parameters["W2"]))

#print("b2 = " + str(parameters["b2"]))
def predict(parameters, X):

       

    A2, cache = forward_propagation(X_test, parameters)

    prediction=np.around(A2,decimals=0)

    prediction=np.argmax(prediction, axis=0)

    

    #Create file with results

    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": prediction})

    submission.to_csv("submission_titanic_set4.csv", index=False)

    print("Prediction_test_test set:", prediction.shape[0])

 

    print(submission.head())

    

    return prediction.T
prediction=print(predict(parameters, X_test))

print(os.listdir("../working"))