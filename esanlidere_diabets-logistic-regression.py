# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

df.head()
df.isnull().sum()
df.info()
df.describe().T
y = df.Outcome.values

X_ = df.drop(["Outcome"], axis = 1)
X = (X_ - np.min(X_)) / (np.max(X_) - np.min(X_)).values #bagimsiz degiskenleri donusturduk
# stats models aracılığıyla değişkenlerin anlamlılık derecelerine bir bakalım

import statsmodels.api as sm

lj = sm.Logit(y, X)

lj_model = lj.fit()

lj_model.summary()
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
X_train = X_train.T

X_test = X_test.T

y_train = y_train.T

y_test = y_test.T
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b
def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b
def sigmoid(z):

    y_pred = 1 / (1 + np.exp(-z))

    return y_pred
def forward_backward_propagation(w,b,X_train,y_train):

    z = np.dot(w.T,X_train) + b

    y_pred = sigmoid(z)

    loss = -y_train*np.log(y_pred)-(1-y_train)*np.log(1-y_pred)

    cost = (np.sum(loss))/X_train.shape[1]

    derivative_weight = (np.dot(X_train,((y_pred-y_train).T)))/X_train.shape[1]

    derivative_bias = np.sum(y_pred-y_train)/X_train.shape[1]          

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients
def update(w, b, X_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    for i in range(number_of_iterarion):

        cost,gradients = forward_backward_propagation(w,b,X_train,y_train)

        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))            

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list
def predict(w,b,X_test):

    z = sigmoid(np.dot(w.T,X_test)+b)

    Y_prediction = np.zeros((1,X_test.shape[1]))

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction
def logistic_regression(X_train, y_train, X_test, y_test, learning_rate ,  num_iterations):

    dimension =  X_train.shape[0] 

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, X_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],X_test)

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

logistic_regression(X_train, y_train, X_test, y_test,learning_rate = 1, num_iterations = 500) 
#sklearn ile LogisticRegression

loj = LogisticRegression(solver = "liblinear")

loj_model =loj.fit(X,y)
loj_model.intercept_  #sabit sayi
loj_model.coef_  #bagimsiz degisken katsayilari
#predict

y_pred = loj_model.predict(X)
#karmasiklik matrisimiz

confusion_matrix(y,y_pred)
accuracy_score(y,y_pred)
# train-test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
loj = LogisticRegression(solver = "liblinear")  # küçük çaplı datasetlerde solver argümanı "liblinear" olduğu zaman daha iyi sonuçlar vermektedir

loj_model = loj.fit(X_train,y_train)
accuracy_score(y_test, loj_model.predict(X_test))