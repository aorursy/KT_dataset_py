# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import plotly.graph_objs as go



from collections import Counter



import warnings

warnings.filterwarnings("ignore")





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/machine-learning-for-diabetes-with-python/diabetes_data.csv")
data.head()
data.info()
data.describe()
y_data = data.Outcome.values

y_data

x_data = data.drop(["Outcome"],axis=1)

x_data
x_data = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

x_data
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,test_size = 0.2, random_state = 10)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T





x_test
x_data.boxplot(column=["BloodPressure","SkinThickness","BMI"])
x_data.boxplot(column=["Pregnancies"])
x_data.boxplot(column=["Glucose"])


def detect_outliers(x_data,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(x_data[c],25)

        # 3rd quartile

        Q3 = np.percentile(x_data[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indeces

        outlier_list_col = x_data[(x_data[c] < Q1 - outlier_step) | (x_data[c] > Q3 + outlier_step)].index

        # store indeces

        outlier_indices.extend(outlier_list_col)

    

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i, v in outlier_indices.items() if v > 1)

    

    return multiple_outliers

x_data.loc[detect_outliers(data,["BloodPressure","SkinThickness","Glucose","BMI"])]

x_data = x_data.drop(detect_outliers(x_data,["BloodPressure","SkinThickness","Glucose","BMI"]),axis = 0).reset_index(drop = True)



def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b



def sigmoid(z):

    

    y_head = 1 / (1+np.exp(-z))

    

    return y_head
def forward_backward_propagation(w,b,x_train,y_head):

    

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss)) / x_train.shape[1]

    

    #backward propogation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
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

            print ("Cost after iteration %i: %f" %(i, cost)) #if section defined to print our cost values in every 10 iteration. We do not need to do that. It's optional.

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iteration")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list


def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is one means has diabete (y_head=1),

    # if z is smaller than 0.5, our prediction is zero means does not have diabete (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]

    w,b = initialize_weights_and_bias(dimension)

    

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    



    # Print train/test Errors

    

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 2, num_iterations =400)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("Test Accuracy {}".format(lr.score(x_test.T,y_test.T)))