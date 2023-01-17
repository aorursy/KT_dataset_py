



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# The classification goal is to predict whether the patient has 10-year risk of future coronary heart disease (CHD).

#The dataset provides the patients’ information

heartDiseaseData = pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")

heartDiseaseData.info()
#heartDiseaseData.drop(["education"], axis = 1, inplace = True) # Delete useless feature.

heartDiseaseData.dropna(how="any", inplace = True)  # Delete useless raw



chd = heartDiseaseData.TenYearCHD.values 

featurees = heartDiseaseData.drop(["TenYearCHD"], axis = 1) # features before normalization

featurees
features = (featurees - np.min(featurees))/(np.max(featurees) - np.min(featurees)).values # features after normalization

features
from sklearn.model_selection import train_test_split

features_train, features_test, chd_train, chd_test = train_test_split(features, chd ,test_size = 0.2 , random_state = 42)

# test_size ==> 80% of data set for Train, 20% of data set for Test





features_train = features_train.T

features_test = features_test.T

chd_train = chd_train.T

chd_test = chd_test.T



print("Changed of Features and Values place.")



print("features_train: ", features_train.shape)

print("features_test ", features_test.shape)

print("chd_train: ", chd_train.shape)

print("chd_test: ", chd_test.shape)
def initialize_weights_and_bias(dimension):

    

    weights = np.full((dimension,1), 0.01) 

    bias = 0.0 

    return weights,bias

    

def sigmoid(z):

    chd_head = 1/(1+np.exp(-z))

    return chd_head
def forward_backward_propagation(weights, bias , features_train, chd_train):

    #forward propagation

    

    z = np.dot(weights.T,features_train) + bias

    chd_head = sigmoid(z)

    loss = -chd_train*np.log(chd_head) - (1- chd_train)*np.log(1-chd_head)

    cost = (np.sum(loss))/features_train.shape[1]

    

    #backward propagation

    derivative_weights = (np.dot(features_train,((chd_head-chd_train).T)))/features_train.shape[1] 

    derivative_bias = np.sum(chd_head-chd_train)/features_train.shape[1] 

    gradients = {"derivative_weights" : derivative_weights, "derivative_bias" : derivative_bias}

    return cost,gradients
def update(weights, bias, features_train, chd_train, learning_rate, number_of_iterations):

    cost_list = []

    cost_list2 = []

    index = []

    

    for i in range(number_of_iterations):

        

        cost, gradients = forward_backward_propagation(weights, bias, features_train, chd_train)

        cost_list.append(cost)

        

        weights = weights - learning_rate* gradients["derivative_weights"]

        bias = bias - learning_rate*gradients["derivative_bias"]

        

        if i % 10 == 0: # her 10 adımda bir depolar

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iterations %i: %f " %(i,cost))

            

        

    parameters =  {"weights" : weights, "bias" : bias}

    plt.plot(index,cost_list2)

    plt.xticks(index, rotation = "vertical")

    plt.xlabel("Number Of Iterations")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(weights, bias, features_test):

    

    z = sigmoid(np.dot(weights.T,features_test)+bias)

    chd_prediction = np.zeros((1,features_test.shape[1]))

    

    

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5 :

            chd_prediction[0,i] = 0

        else:

            chd_prediction[0,i] = 1

            

    return chd_prediction
def logistic_regression(features_train, chd_train, features_test, chd_test, learning_rate, number_of_iterations):

    dimension = features_train.shape[0] # that is 14(features)

    weights, bias = initialize_weights_and_bias(dimension)

    

    parameters, gradients, cost_list = update(weights, bias, features_train, chd_train, learning_rate, number_of_iterations) 

    

    chd_prediction_test = predict(parameters["weights"], parameters["bias"], features_test)

    

    print("Test occuracy: {}% ".format(100-np.mean(np.abs(chd_prediction_test - chd_test))*100))

    

logistic_regression(features_train, chd_train, features_test, chd_test, learning_rate = 5, number_of_iterations = 300) 
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(features_train.T,chd_train.T)

print("test accuracy {}".format(lr.score(features_test.T,chd_test.T)))