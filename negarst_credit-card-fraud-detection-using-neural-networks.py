# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        creditcard_dataset_path = os.path.join(dirname, filename)

        creditcard_data = pd.read_csv(creditcard_dataset_path)

        

# Any results you write to the current directory are saved as output.

print("Number of transactions: ", len(creditcard_data))



creditcard_data.tail()
from sklearn import preprocessing



y = creditcard_data['Class']



# Time column is dropped, because it might be not helpful

X = creditcard_data.drop(['Time', 'Class'], axis = 1)



min_max_scaler = preprocessing.MinMaxScaler()

X = min_max_scaler.fit_transform(X, y)

print(X)



from sklearn.model_selection import train_test_split



# By now, all features are included.

# important_features = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# X = creditcard_data[important_features]

# y = creditcard_data['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Backup for under/over-sampling

X_train_original = X_train

y_train_original = y_train



def reshape_X_sample(X_sample):

    X_reshaped = X_sample.T

    return X_reshaped

def reshape_y_sample(y_sample):

    y_reshaped = y_sample.values.reshape(1, y_sample.shape[0])

    return y_reshaped

    

X_train = reshape_X_sample(X_train)

y_train = reshape_y_sample(y_train)

print ('Train X Shape: ', X_train.shape)

print ('Train Y Shape: ', y_train.shape)

    

X_test = reshape_X_sample(X_test)

y_test = reshape_y_sample(y_test)

print ('\nTest X Shape: ', X_test.shape)
def define_structure(X, Y):

    input_unit = X.shape[0] # size of input layer (30)

    hidden_unit = X.shape[0] #hidden layer of size 30

    output_unit = 1 # size of output layer

    return (input_unit, hidden_unit, output_unit)

(input_unit, hidden_unit, output_unit) = define_structure(X_train, y_train)

print("The size of the input layer (features) is:  = " + str(input_unit))

print("The size of the hidden layer is:  = " + str(hidden_unit))

print("The size of the output layer is:  = " + str(output_unit))
def parameters_initialization(input_unit, hidden_unit, output_unit):

    np.random.seed(32) 

    

    W1 = np.random.randn(hidden_unit, input_unit)*0.01

    #A thirty-by-thirty array of sample from N(mu = 0, sigma^2 = (0.01)^2) 

    

    b1 = np.zeros((hidden_unit, 1))

    #A thirty-by-one array of zeros

    

    W2 = np.random.randn(output_unit, hidden_unit)*0.01

    #A one-by-thirty sample from N(mu = 0, sigma^2 = (0.01)^2) 

    

    b2 = np.zeros((output_unit, 1))

    #A one-by-one array of zeros

    

    parameters = {"W1": W1,

                  "b1": b1,

                  "W2": W2,

                  "b2": b2}

    

    return parameters
def sigmoid(z):

    return 1/(1+np.exp(-z))



def forward_propagation(X, parameters):

    #Initial variables:

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

    #Operational variables:

    Z1 = np.dot(W1, X) + b1

    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2

    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}

    return A2, cache
def cross_entropy_cost(A2, Y, parameters):

    # number of training example

    m = Y.shape[1] 

    # Compute the cross-entropy cost

    logprobs = np.multiply(np.log(A2), 0.1*Y) + np.multiply((1 - 0.1*Y), np.log(1 - A2))

    cost = - np.sum(logprobs) / m

    cost = float(np.squeeze(cost))

                                    

    return cost
def backward_propagation(parameters, cache, X, Y):

    #number of training example

    m = X.shape[1]

    

    W1 = parameters['W1']

    W2 = parameters['W2']

    A1 = cache['A1']

    A2 = cache['A2']

   

    dZ2 = A2-Y

    dW2 = (1/m) * np.dot(dZ2, A1.T)

    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))

    dW1 = (1/m) * np.dot(dZ1, X.T) 

    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

    

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2,"db2": db2}

    

    return grads
def gradient_descent(parameters, grads, learning_rate = 0.1):

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']

   

    dW1 = grads['dW1']

    db1 = grads['db1']

    dW2 = grads['dW2']

    db2 = grads['db2']

    W1 = W1 - learning_rate * dW1

    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2

    b2 = b2 - learning_rate * db2

    

    parameters = {"W1": W1, "b1": b1,"W2": W2,"b2": b2}

    

    return parameters
def neural_network_model(X, Y, hidden_unit, num_iterations = 1000):

    np.random.seed(3)

    input_unit = define_structure(X, Y)[0]

    output_unit = define_structure(X, Y)[2]

    

    parameters = parameters_initialization(input_unit, hidden_unit, output_unit)

   

    W1 = parameters['W1']

    b1 = parameters['b1']

    W2 = parameters['W2']

    b2 = parameters['b2']



    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)

        cost = cross_entropy_cost(A2, Y, parameters)

        grads = backward_propagation(parameters, cache, X, Y)

        parameters = gradient_descent(parameters, grads)

        if i % 5 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters

parameters = neural_network_model(X_train, y_train, 29, num_iterations=500)
def prediction(parameters, X):

    A2, cache = forward_propagation(X, parameters)

    predictions = np.round(A2)

    

    return predictions



predictions = prediction(parameters, X_train)

print ('Accuracy Train: %d' % float((np.dot(y_train, predictions.T) + np.dot(1 - y_train, 1 - predictions.T))/float(y_train.size)*100) + '%')

predictions = prediction(parameters, X_test)

print ('Accuracy Test: %d' % float((np.dot(y_test, predictions.T) + np.dot(1 - y_test, 1 - predictions.T))/float(y_test.size)*100) + '%')
def report_accuracy(predictions, y_test):

    predicted_fraud_as_fraud_truely = 0

    predicted_fraud_as_nonfraud_falsely = 0

    predicted_y = [int(i) for i in predictions[0]]



    for i in range(len(predicted_y)):

        if predicted_y[i] == 1 and y_test[0][i] == 1:

            predicted_fraud_as_fraud_truely = predicted_fraud_as_fraud_truely + 1

        if predicted_y[i] == 0 and y_test[0][i] == 1:

            predicted_fraud_as_nonfraud_falsely = predicted_fraud_as_nonfraud_falsely + 1

    print('predicted_fraud_as_fraud_truely: ', predicted_fraud_as_fraud_truely)

    print('predicted_fraud_as_nonfraud_falsely: ', predicted_fraud_as_nonfraud_falsely)

    print(predicted_fraud_as_fraud_truely / (predicted_fraud_as_fraud_truely + predicted_fraud_as_nonfraud_falsely) * 100, " Of true frauds has been predicted as fraud.")

    

report_accuracy(predictions, y_test)
# colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y]

# kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}

# fig = plt.Figure(figsize=(12,6))

# plt.scatter(X[:, 0], X[:, 1], c=colors, **kwarg_params)

# sns.despine()
# from imblearn.over_sampling import RandomOverSampler



# ros = RandomOverSampler(random_state=0)

# X_resampled, y_resampled = ros.fit_sample(X_train_original, y_train_original)

# colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled]

# plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='black')

# sns.despine()

# plt.title("RandomOverSampler Output ($n_{class}=4700)$")

# pass

from imblearn.over_sampling import RandomOverSampler



print('Number of frauds in the train dataset:', sum(y_train_original))

print('Number of non-frauds in the train dataset:', len(y_train_original) - sum(y_train_original))

print('Fraud ratio in the train dataset:', sum(y_train_original) / (len(y_train_original) - sum(y_train_original)))



print('Over-sampling...')



ros = RandomOverSampler(0.20, random_state=0)

X_res, y_res = ros.fit_resample(X_train_original, y_train_original)



print('Number of frauds in the train dataset:', sum(y_res))

print('Number of non-frauds in the train dataset:', len(y_res) - sum(y_res))

print('Fraud ratio in the train dataset:', sum(y_res) / (len(y_res) - sum(y_res)))



# colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled]

# plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='black')

# sns.despine()

# plt.title("RandomOverSampler Output ($n_{class}=4700)$")

# pass

# print(X_res)

# Convert the resampler output list of lists to a dataframe.

X_df = pd.DataFrame(list(map(np.ravel, X_res)))

print(X_df)
X_train = reshape_X_sample(X_df)

y_train = y_res.reshape(1, y_res.shape[0])

parameters = neural_network_model(X_train, y_train, 29, num_iterations=3000)

predictions = prediction(parameters, X_train)

print ('Accuracy Train: %d' % float((np.dot(y_train, predictions.T) + np.dot(1 - y_train, 1 - predictions.T))/float(y_train.size)*100) + '%')

predictions = prediction(parameters, X_test)

print ('Accuracy Test: %d' % float((np.dot(y_test, predictions.T) + np.dot(1 - y_test, 1 - predictions.T))/float(y_test.size)*100) + '%')

report_accuracy(predictions, y_test)
from imblearn.under_sampling import RandomUnderSampler



print('Number of frauds in the train dataset:', sum(y_train_original))

print('Number of non-frauds in the train dataset:', len(y_train_original) - sum(y_train_original))

print('Fraud ratio in the train dataset:', sum(y_train_original) / (len(y_train_original) - sum(y_train_original)))



print('Under-sampling...')



rus = RandomUnderSampler(0.20, random_state=0)

X_res, y_res = rus.fit_resample(X_train_original, y_train_original)



print('Number of frauds in the train dataset:', sum(y_res))

print('Number of non-frauds in the train dataset:', len(y_res) - sum(y_res))

print('Fraud ratio in the train dataset:', sum(y_res) / (len(y_res) - sum(y_res)))





# colors = ['#ef8a62' if v == 0 else '#f7f7f7' if v == 1 else '#67a9cf' for v in y_resampled]

# plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=colors, linewidth=0.5, edgecolor='black')

# sns.despine()

# plt.title("RandomUnderSampler Output ($n_{class}=64)$")

# pass
# print(X_res)

# Convert the resampler output list of lists to a dataframe.

X_df = pd.DataFrame(list(map(np.ravel, X_res)))

print(X_df)
X_train_rus = reshape_X_sample(X_df)

y_train_rus = y_res.reshape(1, y_res.shape[0])

parameters_rus = neural_network_model(X_train_rus, y_train_rus, 29, num_iterations=3000)



predictions_rus = prediction(parameters_rus, X_train_rus)

print ('Accuracy Train: %d' % float((np.dot(y_train_rus, predictions_rus.T) + np.dot(1 - y_train_rus, 1 - predictions_rus.T))/float(y_train_rus.size)*100) + '%')

predictions_rus = prediction(parameters_rus, X_test)

print ('Accuracy Test: %d' % float((np.dot(y_test, predictions_rus.T) + np.dot(1 - y_test, 1 - predictions_rus.T))/float(y_test.size)*100) + '%')

report_accuracy(predictions_rus, y_test)
from sklearn.metrics import confusion_matrix

predicted_y = np.array([int(i) for i in predictions_rus[0]])



print("Confusion Matrix:::")

print(confusion_matrix(y_test[0], predicted_y))



tn, fp, fn, tp = confusion_matrix(y_test[0], predicted_y).ravel()

print("\nTN: ", tn,

      "\nFP: ", fp,

      "\nFN: ", fn,

      "\nTP: ", tp)