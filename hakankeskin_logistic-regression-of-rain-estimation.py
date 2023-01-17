# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

# filter warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")

df.head()

df.info()


df.drop(["Date", "Location","WindDir9am","WindDir3pm", "WindGustDir", "RainToday"], axis = 1, inplace = True)

df.dropna(inplace=True)

df.info()

df.RainTomorrow = [1 if each == "Yes" else 0 for each in df.RainTomorrow]

df.RainTomorrow.value_counts()
df.info()
y = df.RainTomorrow.values

x_data = df.drop(["RainTomorrow"], axis = 1)

# Lets normalize the dataset to pretend higher scaling factor between data features.



x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)



x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T



print("x_train:", x_train.shape)

print("y_train:", y_train.shape)

print("x_test:", x_test.shape)

print("y_test:", y_test.shape)
def initialize_weigths_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b



w,b = initialize_weigths_and_bias(5) # example

def sigmoid(z):

    y_head = 1/(1 + np.exp(-z))

    return y_head



y_head = sigmoid(0.9) # example

y_head
def forward_backward_propagation(w,b,x_train,y_train):

    #forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]              # x_train.shape[1] is for scailing

    

    #backward propagation

    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    return cost,gradients



def update(w,b,x_train,y_train,learning_rate, number_of_iteration):

    cost_list = []

    cost_list2 = []

    index = []

    

    #updating learning parameters is number of iteration times

    

    for i in range(number_of_iteration):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        

        #lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b- learning_rate * gradients["derivative_bias"]

        

        if i % 10  == 0:

            cost_list2.append(cost)

            index.append(i)

            print("Cost after iteration %i: %f" %(i,cost))

    # we update(learn) parameters weigths and bias

    parameters = {"weight": w, "bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



def predict(w,b,x_test):

    

    z = sigmoid(np.dot(w.T,x_test)+b)

    y_predict = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i] <= 0.5:

            y_predict[0,i] = 0

        else:

            y_predict[0,i] = 1

            

    return y_predict
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):

    dimension = x_train.shape[0]

    w,b = initialize_weigths_and_bias(dimension)

    

    parameters, gradients, cost_list = update(w,b,x_train,y_train,learning_rate,num_iterations)

    y_predict_test = predict(parameters["weight"], parameters["bias"], x_test)

    

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict_test - y_test)) * 100))

    

logistic_regression(x_train,y_train,x_test,y_test,learning_rate = 3,num_iterations=1000)


from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()



lr.fit(x_train.T, y_train.T)



print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
# KNN Algorithm of Data



x_train = x_train.T

y_train = y_train.T

x_test = x_test.T

y_test = y_test.T
print("x_train:", x_train.shape)

print("y_train:", y_train.shape)

print("x_test:", x_test.shape)

print("y_test:", y_test.shape)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 25)



knn.fit(x_train,y_train)

prediction = knn.predict(x_test)



print(" {} nn score: {}".format(25,knn.score(x_test,y_test)))



# How can we find best number of neighbors with for loop



score_list = []



for each in range(1,25):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,25), score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
