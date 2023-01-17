import numpy as np

import pandas as pd

import tqdm
data_train = pd.read_csv('../input/titanic/train.csv')

data_test = pd.read_csv('../input/titanic/test.csv')

print(data_train.shape)

print(data_test.shape)
data_train.head()
data_train.isnull().sum()
data_test.isnull().sum()
#function that replaces missing data based on the method and key parameters

def fill_nan(data, key, method = "mean"):

    if method == "mean":

        data[key].fillna(data["Age"].mean(), inplace = True)

    if method == "mode":

        data[key].fillna(data["Age"].mode()[0], inplace = True)

    if method == "median":

        data[key].fillna(data["Age"].median(), inplace = True)
#make copys of our data. deep = true means that each entry of our data also gets copied to a different memory address

data_train_cleaned = data_train.copy(deep = True)

data_test_cleaned = data_test.copy(deep = True)



#calculate stats of our data

data_train_cleaned.describe(include = 'all')

data_test_cleaned.describe(include = 'all')



#clean data

#fill empty age

fill_nan(data_train_cleaned, "Age", "median")

fill_nan(data_test_cleaned, "Age", "median")



#fill empty embarked in train

data_train_cleaned["Embarked"].fillna(data_train_cleaned["Embarked"].mode()[0], inplace = True)



#fill empty fare in test

data_test_cleaned["Fare"].fillna(data_test_cleaned["Fare"].mean(), inplace = True)
data_train_cleaned = data_train_cleaned.drop("Cabin", axis = 1)

data_test_cleaned = data_test_cleaned.drop("Cabin", axis = 1)
data_train_cleaned = data_train_cleaned.drop(["PassengerId", "Name", "Ticket"], axis = 1)

data_test_cleaned = data_test_cleaned.drop(["PassengerId", "Name", "Ticket"], axis = 1)
print(data_train_cleaned.head())

print(data_train_cleaned.isnull().sum())

print(data_test_cleaned.isnull().sum())
#map Sex of a passenger to interger values , female : 0 , male : 1

data_train_cleaned['Sex'] = data_train_cleaned['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

data_test_cleaned['Sex'] = data_test_cleaned['Sex'].map( {'female': 0, 'male': 1} ).astype(int)



#map embarked of a passenger to integer values S: 0, C : 1, Q : 2

data_train_cleaned['Embarked'] = data_train_cleaned['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)

data_test_cleaned['Embarked'] = data_test_cleaned['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)



#visualize data

print(data_train_cleaned.head())

print(data_test_cleaned.head())
#make a copy of our data to slice it

X_data = data_train_cleaned.copy(deep = True).values # .values converst pandas dataframe to a numpy array



#split data into train and val

X_train = X_data[:623] #70% of our training data 891 is ~ 623 values

X_val = X_data[623:] #30% of our training data is ~ 268 values



# labels are " survived " column of the dataset

Y_train = X_train[:,0]

Y_val = X_val[:,0]



#remove labels from dataset and only keep features

X_train = np.delete(X_train, 0, axis = 1)

X_val = np.delete(X_val, 0, axis = 1)



#print data for sanity check

print("x train : " + str(X_train))

print("x val : " + str(X_val))

print("y train:"+ str(Y_train))

print("y val : " +str(Y_val))



#print shapes for sanity check

print("x train :" + str(X_train.shape))

print("x val :" + str(X_val.shape))

print("y train :" + str(Y_train.shape))

print("y val :" + str(Y_val.shape) )
# rearrange data such that each column is a different training example

X_train = X_train.T

X_val = X_val.T



#fix our lable matrix

Y_train = Y_train.reshape((Y_train.shape[0], 1))

Y_val = Y_val.reshape((Y_val.shape[0], 1))



#sanity check

print("x train :" + str(X_train.shape))

print("x val :" + str(X_val.shape))

print("y train :" + str(Y_train.shape))

print("y val :" + str(Y_val.shape) )
#import matplotlib to plot our sigmoid function

%matplotlib inline

import matplotlib.pyplot as plt

x = np.linspace(-7, 7, 200)



#define sigmoid function

def sigmoid(x):

    sig = 1/(1 + np.exp(-x))

    return sig



plt.plot(x, sigmoid(x))
# inorder to standardize our dataset , we will subtract all features of our dataset by their mean

# and divide it by the standard deviation of the features.

# this will center the data and normalize our values so that gradient descent converges faster !

def calc_stats(data):

    mu = data.mean(axis = 1, keepdims = True)

    sigma = data.std(axis = 1, keepdims = True)

    return mu, sigma



def standardize(data, mu, sigma):

    std_data = (data - mu) / sigma

    return std_data

mu, sigma = calc_stats(X_train)

X_train = standardize(X_train, mu, sigma)

X_val = standardize(X_val, mu, sigma)



#sanity check !

print(X_train.shape)

print(X_train[:5])
# initialize parameters

def initialize_parameters(dim):

    W = np.zeros((dim, 1))

    b = 0

    return W, b
#forward propagation

def forward_prop(X, W, b):

#     #sanity check

#     print("forward prop")

#     print("X shape:" + str(X.shape))

#     print("W shape:" + str(W.shape))

    Z = np.dot(W.T, X) + b

    A = sigmoid(Z)

    return A.T
#cost function

def compute_cost(Y, A):

#     #sanity check

#     print("computing cost")

#     print("Y shape:" + str(Y.shape))

#     print("A shape:" + str(A.shape))

    J = - np.sum(np.dot(Y.T, np.log(A)) + np.dot((1 - Y).T, np.log(1 - A)))/Y.shape[0]

    return J
# back prop function

def back_prop(X, Y, A):

    #sanity check

#     print("back_prop")

#     print("X shape:" + str(X.shape))

#     print("Y shape:" + str(Y.shape))

#     print("A shape:" + str(A.shape))

    dW = np.dot(X, (A-Y))/X.shape[1]

    db = np.sum(A - Y)/X.shape[1]

    

    return dW, db
#gradient descent

def gradient_descent(W, b, dW, db, learning_rate = 0.001):

    W = W - learning_rate * dW

    b = b - learning_rate * db

    return W, b
# Logistic regression function !

def logistic_regression(X, Y, num_iterations, learning_rate, print_cost = False, cost_graph = False):

    m = X_train.shape[1] #number of training examples

    W, b = initialize_parameters(X_train.shape[0]) #initialize learning parameters

    for i in tqdm.tqdm(range(num_iterations)):

        

        A = forward_prop(X, W, b)

        cost = compute_cost(Y, A)

        dW, db = back_prop(X, Y, A)

        W, b = gradient_descent(W, b, dW, db, learning_rate)

        

        # Record the costs

        if i % 100 == 0:

            costs.append(cost)

        # Print the cost every 100 training iterations

        if print_cost and i % 100 == 0:

            print("Cost after iteration %i: %f" %(i, cost))    

    if cost_graph == True:

        plt.plot(costs)

    return W, b
#make predictions !

def predict(X_val, W, b):

    predictions = forward_prop(X_val, W, b)

    

    #map predictions below 0.5 to 0 and above 0.5 to 1

    predictions[predictions > 0.5] = int(1)

    predictions[predictions < 0.5] = int(0)

    return predictions



# calculate accuracy

def test_accuracy(predictions, Y_val):

    accuracy = np.sum(predictions == Y_val)/predictions.shape[0]*100

    return accuracy
costs = [] #store cost to plot against iterations

W, b = logistic_regression(X_train, Y_train, 10000, 0.01, cost_graph = True)
#make predictions on our validation dataset

preds_train = predict(X_train, W, b)

preds_val = predict(X_val, W, b)

#calculate accuracy of our predictions

print(f"Accuracy on train data {test_accuracy(preds_train, Y_train)}%")

print(f"Accuracy on validation data {test_accuracy(preds_val, Y_val)}%")
#import test dataset

X_test = data_test_cleaned.values



#standardize test dataset

X_test = X_test.T

X_test = standardize(X_test, mu, sigma)



#make predictions

predictions_test = predict(X_test, W, b).astype(int)



#compile into a dataframe

predictions_df = pd.DataFrame({ 'PassengerId': data_test["PassengerId"], 'Survived': predictions_test[:,0]})



#export dataframe as csv

predictions_df.to_csv("../working/Logistic_regression.csv", index = False)



predictions_df