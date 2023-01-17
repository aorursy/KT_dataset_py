from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.models import Sequential

import pandas as pd 

import numpy as np 

import keras

import missingno
dataset = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")

dataset.head()
#Yes and No labelled to 1 and 0; RainTomorrow and RainToday column

dataset.loc[dataset['RainTomorrow'] == "Yes", 'RainTomorrow'] = 1.0

dataset.loc[dataset['RainTomorrow'] == "No", 'RainTomorrow'] = 0.0

dataset.loc[dataset['RainToday'] == "Yes", 'RainToday'] = 1.0

dataset.loc[dataset['RainToday'] == "No", 'RainToday'] = 0.0

missingno.matrix(dataset, figsize = (30, 10))         #Total missing values show

dataset.isnull().sum()
dataset["MinTemp"] = dataset["MinTemp"].fillna(0)

dataset["MaxTemp"] = dataset["MaxTemp"].fillna(0)

dataset["Rainfall"] = dataset["Rainfall"].fillna(0)

dataset["Evaporation"] = dataset["Evaporation"].fillna(0)

dataset["Sunshine"] = dataset["Sunshine"].fillna(0)

dataset["WindGustDir"] = dataset["WindGustDir"].fillna(0)

dataset["WindGustSpeed"] = dataset["WindGustSpeed"].fillna(0)

dataset["WindDir9am"] = dataset["WindDir9am"].fillna(0)

dataset["WindDir3pm"] = dataset["WindDir3pm"].fillna(0)

dataset["WindSpeed9am"] = dataset["WindSpeed9am"].fillna(0)

dataset["WindSpeed3pm"] = dataset["WindSpeed3pm"].fillna(0)

dataset["Humidity9am"] = dataset["Humidity9am"].fillna(0)

dataset["Humidity3pm"] = dataset["Humidity3pm"].fillna(0)

dataset["Pressure9am"] = dataset["Pressure9am"].fillna(0)

dataset["Pressure3pm"] = dataset["Pressure3pm"].fillna(0)

dataset["Cloud9am"] = dataset["Cloud9am"].fillna(0)

dataset["Cloud3pm"] = dataset["Cloud3pm"].fillna(0)

dataset["Temp9am"] = dataset["Temp9am"].fillna(0)

dataset["Temp3pm"] = dataset["Temp3pm"].fillna(0)

dataset["RainToday"] = dataset["RainToday"].fillna(0)
# include all columns

X = dataset



# target variable; only RainTomorrow column

Y = dataset[['RainTomorrow']]



# remove RainTomorrow, WindGustDir, WindDir9am,WindDir3pm Location, Date from X

X = X.drop(['RainTomorrow'], axis = 1) 

X = X.drop(['WindGustDir','WindDir9am','WindDir3pm','Location', 'Date'], axis = 1)



# Spliting the dataset into test and train set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

Y.head()
X_train.head()
Y_train.head()
def sigmoid(z):

    s = 1.0/ (1.0 + np.exp(-z))    

    return s
def network_architecture(X, Y):

    # nodes in input layer

    n_x = X.shape[0] 

    # nodes in hidden layer

    n_h = 10          

    # nodes in output layer

    n_y = Y.shape[0] 

    return (n_x, n_h, n_y)
def define_network_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h,n_x) * 0.01 # random initialization

    b1 = np.zeros((n_h, 1)) # zero initialization

    W2 = np.random.randn(n_y,n_h) * 0.01 

    b2 = np.zeros((n_y, 1)) 

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}  
def forward_propagation(X, params):

    Z1 = np.dot(params['W1'], X)+params['b1']

    A1 = sigmoid(Z1)



    Z2 = np.dot(params['W2'], A1)+params['b2']

    A2 = sigmoid(Z2)

    return {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2} 
def compute_error(Predicted, Actual):

    logprobs = np.multiply(np.log(Predicted), Actual)+ np.multiply(np.log(1-Predicted), 1-Actual)

    cost = -np.sum(logprobs) / Actual.shape[1] 

    return np.squeeze(cost)
def backward_propagation(params, activations, X, Y):

    m = X.shape[1]

    

    # output layer

    dZ2 = activations['A2'] - Y # compute the error derivative 

    dW2 = np.dot(dZ2, activations['A1'].T) / m # compute the weight derivative 

    db2 = np.sum(dZ2, axis=1, keepdims =True)/m # compute the bias derivative #keepdims = True i.e. input output dimension same thakbe

    

    # hidden layer

    dZ1 = np.dot(params['W2'].T, dZ2)*(1-np.power(activations['A1'], 2))

    dW1 = np.dot(dZ1, X.T)/m

    db1 = np.sum(dZ1, axis=1,keepdims=True)/m

    

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}



def update_parameters(params, derivatives, alpha = 1.2):

    # alpha is the model's learning rate 

    

    params['W1'] = params['W1'] - alpha * derivatives['dW1']

    params['b1'] = params['b1'] - alpha * derivatives['db1']

    params['W2'] = params['W2'] - alpha * derivatives['dW2']

    params['b2'] = params['b2'] - alpha * derivatives['db2']

    return params
def neural_network(X, Y, n_h, num_iterations=100):

    n_x = network_architecture(X, Y)[0]

    n_y = network_architecture(X, Y)[2]

    

    params = define_network_parameters(n_x, n_h, n_y)

    for i in range(0, num_iterations):

        results = forward_propagation(X, params)

        error = compute_error(results['A2'], Y)

        derivatives = backward_propagation(params, results, X, Y) 

        params = update_parameters(params, derivatives)    

    return params
X = X_train

Y = Y_train

y = Y.values.reshape(1, Y.size)

x = X.T.as_matrix()
model = neural_network(x, y, n_h = 10, num_iterations = 100)
def predict(parameters, X):

    results = forward_propagation(X, parameters)

    print (results['A2'][0])

    predictions = np.around(results['A2'])    

    return predictions



predictions = predict(model, x)

print ('Accuracy: %d' % float((np.dot(y,predictions.T) + np.dot(1-y,1-predictions.T))/float(y.size)*100) + '%')