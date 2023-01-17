# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# reads the data

data = pd.read_csv("../input/data.csv")
data.info()
# drops the unused columns

data.drop(["id", "Unnamed: 32"], axis=1, inplace = True)



# drops the column which is the feature to be predicted

x_data = data.drop(["diagnosis"], axis=1)
# converts the non-numeric values to numeric values in the column

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
#normalizes the x axis and assigns x and y axis

x = ((x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))).values

y = data.diagnosis.values
# splits the data as train and test parts

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T



def initialize_parameters(dimension):

    theta0 = 0.0

    thetaLeft = np.full((dimension, 1), 0.01)

    return theta0, thetaLeft
def sigmoid(z):

    return (1/(1+ np.exp(-z)))
def forward_backward_propagation(theta0, thetaLeft, x_train, y_train):

    z = theta0 + np.dot(thetaLeft.T, x_train) 

    y_head = sigmoid(z)

    cost_function = -y_train*np.log(y_head) - (1 - y_head)*np.log(1 - y_head)

    cost = (np.sum(cost_function))/x_train.shape[1]

    

    derivative_theta0 = np.sum(y_head - y_train)/x_train.shape[1]

    derivative_thetaLeft = (np.dot(x_train, ((y_head - y_train).T)))/x_train.shape[1]

    gradients = {"derivative_theta0": derivative_theta0, "derivative_thetaLeft": derivative_thetaLeft}

    return cost, gradients
cost_list = []

cost_list_to_plot = []

index_to_plot = []



def update(theta0, thetaLeft, x_train, y_train, learning_rate, number_of_iteration):



    for i in range(1, number_of_iteration + 1):

        

        cost, gradients = forward_backward_propagation(theta0, thetaLeft, x_train, y_train)

        cost_list.append(cost)

        

        theta0 = theta0 - learning_rate*gradients["derivative_theta0"]

        thetaLeft = thetaLeft - learning_rate*gradients["derivative_thetaLeft"]

        

        if i % 20 == 0:

            cost_list_to_plot.append(cost)

            index_to_plot.append(i)

            # print("Cost after iteration %i: %f" %(i, cost))

        

    parameters = {"theta0": theta0, "thetaLeft": thetaLeft}

        



    return parameters, gradients, cost_list

            
def predict(theta0, thetaLeft, x_test):

    z = sigmoid(theta0 + np.dot(thetaLeft.T, x_test))

    Y_prediction = np.zeros((1, x_test.shape[1]))

    

    for i in range (z.shape[1]):

        if z[0, i] <= 0.5:

            Y_prediction[0, i] = 0

        else:

            Y_prediction[0, i] = 1

            

    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, number_of_iteration):

    dimension = x_train.shape[0]

    theta0, thetaLeft = initialize_parameters(dimension)

    parameters, gradients, cost_list = update(theta0, thetaLeft, x_train, y_train, learning_rate, number_of_iteration)

    

    y_prediction_test = predict(parameters["theta0"], parameters["thetaLeft"], x_test)

    y_prediction_train = predict(parameters["theta0"], parameters["thetaLeft"], x_train)

    

    cost_list_global_train.append(100-np.mean(np.abs(y_prediction_train - y_train))*100)

    cost_list_global_test.append(100-np.mean(np.abs(y_prediction_test - y_test))*100)

    

    print("train accuracy: % {}".format(100 - np.mean(np.abs(y_prediction_train-y_train))*100))

    print("test accuracy: % {}".format(100 - np.mean(np.abs(y_prediction_test-y_test))*100))    
# logistic_regression function with different learning_rates and different iteration times



cost_list_global_train = []

cost_list_global_test = []



iteration_number_list = []

learning_rate_number_list = []

hitbreak = 5 # optional

iter_ = 0

alpha_ = 0.00



for i in range (hitbreak):

    iter_ += 20

    alpha_ = 0.00

    for i in range(hitbreak):

        alpha_ += 0.1

        iteration_number_list.append(iter_)

        learning_rate_number_list.append(round(alpha_, 2))

        logistic_regression(x_train, y_train, x_test, y_test, learning_rate = alpha_, number_of_iteration = iter_)

        

        if len(cost_list_to_plot) == 1:

            plt.plot(index_to_plot, cost_list_to_plot,'.')

        else:

            plt.plot(index_to_plot, cost_list_to_plot)

        

        plt.plot(index_to_plot, cost_list_to_plot)

        plt.xticks(index_to_plot, rotation = 'vertical')

        plt.xlabel("Number of iteration")

        plt.ylabel("Cost")

        plt.figure()

        plt.show()

        

        cost_list = []

        cost_list_to_plot = []

        index_to_plot = []

        

    



# dataframe that used for creating annonated heatmap



data = {"Learning Rate": [each for each in learning_rate_number_list], "Number of Iteration": [each for each in iteration_number_list], "Accuracy":cost_list_global_train}

df_rough = pd.DataFrame(data)

df = df_rough.pivot("Learning Rate", "Number of Iteration", "Accuracy")

# creates heatmap



import seaborn as sns



sns.set(font_scale=2)

labels = (np.asarray(["Train: {0:.5f}\nTest: {1:.5f}".format(symb, value) for symb, value in zip(np.asarray(cost_list_global_train).reshape(hitbreak,hitbreak).T.flatten(), np.asarray(cost_list_global_test).reshape(hitbreak,hitbreak).T.flatten() )])).reshape(hitbreak,hitbreak)

f, ax = plt.subplots(figsize=(35, 20))





sns.heatmap(df, annot=labels, fmt = "", linewidths=.5, ax=ax, annot_kws={"size": 20}, cmap='YlGn')
# logistic regression with sklearn



from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T, y_train.T)

print("test accuracy {}".format(lr.score(x_test.T, y_test.T)))
