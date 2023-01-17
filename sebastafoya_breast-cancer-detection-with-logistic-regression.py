import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

    

#Initialization

def initialize(dimension):

    """

    This function creates a vector of the given initialization value of shape (dim, 1) for w and initializes b to 0.



    Argument:

    dimension -- size of the w vector we want (or number of parameters in this case)

    

    Returns:

    w -- initialized vector of shape (dim, 1)

    b -- initialized scalar (corresponds to the bias)

    """

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b

    

#Sigmoid function

def sigmoid(z):

    """

    Compute the sigmoid of z

    

    Arguments:

    z -- A scalar or numpy array of any size.

    

    Return:

    s -- sigmoid(z)

    """

        

    A = 1/(1+np.exp(-z))

        

    return A

    

def propagate(w,b,X,Y):

    """

    Implement the cost function and its gradient for the previous propagation

    

    Arguments:

    w -- weights, a numpy array of size (number of parameters, 1)

    b -- bias, a scalar

    X -- data of size (number of parameters, number of samples)

    Y -- true "label" vector (containing 0 if benignant, 1 malignant) of size (number of samples, 1)

    

    Return:

    cost -- negative log-likelihood cost for logistic regression

    dw -- gradient of the loss with respect to w, thus same shape as w

    db -- gradient of the loss with respect to b, thus same shape as b

    """

        

    #Forward propagation

    #Linear function

    z = np.dot(w.T,X) + b

        

    #Sigmoid function

    A = sigmoid(z)

        

    #Get the number of samples

    m = X.shape[1]

        

    #Calculate the cost

    loss = Y * np.log(A) + (1-Y) * np.log(1-A)  

    cost = (-1./m) * np.sum(loss)

        

    #Backward propagation

    dw = (1./m) * np.dot(X,(A-Y).T)

    db = (1./m) * np.sum(A-Y)   

        

    gradients = {"dw": dw,"db": db}

        

    return cost,gradients

    

def optimize(w, b, X, Y, learning_rate, epochs):

    """

    This function optimizes w and b by running a gradient descent algorithm

    

    Arguments:

    w -- weights, a numpy array of size (number of parameters, 1)

    b -- bias, a scalar

    X -- data of size (number of parameters, number of samples)

    Y -- true "label" vector (containing 0 if benignant, 1 malignant) of size (number of samples, 1)

    learning_rate -- learning rate of the gradient descent update rule

    epochs -- number of iterations of the optimization loop

    

    Returns:

    parameters -- dictionary containing the weights w and bias b

    gradients -- dictionary containing the gradients of the weights and bias with respect to the cost function

    cost_list -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    """

        

    cost_list = []

    cost_list2 = []

    index = []

        

    for i in range(epochs):

        #Make forward and backward propagation and find cost and gradients

        cost, gradients = propagate(w, b, X, Y)

        cost_list.append(cost)

            

        #Retrieve gradients

        dw = gradients['dw']

        db = gradients['db']

            

        #Update parameters

        w = w - learning_rate * dw

        b = b - learning_rate * db

            

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

                

    parameters = {"weights": w,"bias": b}

        

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

        

    return parameters, gradients, cost_list

    

def predict(w, b, X):

    '''

    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    

    Arguments:

    w -- weights, a numpy array of size (number of parameters, 1)

    b -- bias, a scalar

    X -- data of size (number of parameters, number of samples)

    

    Returns:

    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X

    '''

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

        

    #Compute vector "A" predicting the probabilities of the tumor being malignant

    A = sigmoid(np.dot(w.T, X) + b)

        

    for i in range(A.shape[1]):

            

        if A[0,i] <= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

    

    return Y_prediction

    

def model(X_train, Y_train, X_test, Y_test, learning_rate, epochs = 2000):

    """

    Builds the logistic regression model by calling the function you've implemented previously

    

    Arguments:

    X_train -- training set represented by a numpy array of shape (number of parameters, number of training samples)

    Y_train -- training labels represented by a numpy array (vector) of shape (1, number of training samples)

    X_test -- test set represented by a numpy array of shape (number of parameters, number of testing samples)

    Y_test -- test labels represented by a numpy array (vector) of shape (1, number of testing samples)

    epochs -- hyperparameter representing the number of iterations to optimize the parameters

    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()

    

    Returns:

    d -- dictionary containing information about the model.

    """

        

    #Get number of parameters

    dimension =  x_train.shape[0]

        

    #Initialization

    w,b = initialize(dimension)

        

    #Train the model

    parameters, gradients, cost_list = optimize(w, b, x_train, y_train, learning_rate, epochs)

        

    #Make the predictions

    y_prediction_train = predict(parameters["weights"], parameters["bias"], x_train)

    y_prediction_test = predict(parameters["weights"], parameters["bias"], x_test)

        

    #Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

if __name__ == '__main__':

    

    #Load file to data frame

    df = pd.read_csv("../input/data.csv")

    

    #Explore the file

    print(df.head()) #See the head of the data frame

    print(df.columns) #List its columns

    print(df.describe()) #Statistical descriptions of the columns

    

    #Drop unuseful columns

    df.drop(['Unnamed: 32', 'id'], axis = 1, inplace=True)

    

    #Binarize diagnosis column

    df.diagnosis = [1 if each == 'M' else 0 for each in df.diagnosis]

    

    #Create labels vector

    y = df.diagnosis.values

    

    #Drop diagnosis column and create x data frame

    x_data = df.drop(['diagnosis'], axis=1)

    

    #Check the count for malignant and benignant tumors

    yPlot = df.diagnosis

    ax = sns.countplot(yPlot, label='Count')

    plt.show()

    B, M = yPlot.value_counts()

    print('Number of Benignants:', B)

    print('Number of Malignants', M)

    

    #Normalize the data

    x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

    

    #Split in train and test sets

    from sklearn.model_selection import train_test_split

    (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.15, random_state=42)

    

    x_train = x_train.T

    x_test = x_test.T

    y_train = y_train.T

    y_test = y_test.T

    

    print("x train: ",x_train.shape)

    print("x test: ",x_test.shape)

    print("y train: ",y_train.shape)

    print("y test: ",y_test.shape)

    

    #Run the model

    model(x_train, y_train, x_test, y_test, learning_rate = 1, epochs = 100)