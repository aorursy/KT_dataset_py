# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Import Data

df = pd.read_csv('../input/Iris.csv')
# First 5 rows of Data

df.head()
# Lowercase  all column
df.columns = map(str.lower, df.columns)
# See which classes we have
df.species.unique()
# As we will create model for 2 different classes we drop 'Iris-virginica'

df = df[df.species != 'Iris-setosa'].drop('id',axis=1)
#Summary of data

df.info()
# plot the data with Species Label

sns.pairplot(df, hue='species', size=2.5)
# We need to change Species column data type to categorical to apply .cat.codes method

df.species = df.species.astype('category')

# We assign 0 for Iris-setosa and 1 for Iris-versicolor

df["species"] = df["species"].cat.codes
df.head()
# Create x variable for our feature. Drop species and species_type as they are not features 

x = df.drop('species',axis = 1)

# Create y variable for label of x features

y = df.species.values
# Check x
x.head()
# Whitening Normalization
x_norm = (x-np.mean(x))/np.std(x)
#Normalized x
x_norm.head()
# Split the data

x_train,x_test,y_train,y_test = train_test_split(x_norm, y, test_size=0.2, random_state=42)
# Pandas Dataframe create arrays with features on column and observations on rows.
# We need to transpose these arrays to fit Logistic Regression Equation   
x_test = x_test.T
x_train = x_train.T
# weight initialization function
def weight_init(dimension) :
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b
# We use sigmoid function to have 1-0 prediction.
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
def logistic_forward_propagation(x_train):
    
    dimension =  x_train.shape[0]
    
    w,b = weight_init(dimension)
    
    z = np.dot(w.T,x_train) + b
    
    y_head = sigmoid(z)     
    
    # To calculate how far our prediction from real value we use Log likehood Function
    # Log Likehood = (y * log(y_head)) + ((1−y) * log(1−y_head))
    # Difference between real values and predicted values is Cost Function our aim is to minimize Cost Function
    # Cost Function = ∑-(Log Likehood)
    
    inv_loss = (y_train * np.log(y_head)) + ((1-y_train) * np.log(1-y_head))
    
    cost = (-1/x_train.shape[1]) * (np.sum(inv_loss))
    
    return cost
logistic_forward_propagation (x_train)
def grading_decent(w, b, x_train, learning_rate, iteration_num):
    
    # we use grading desenct to minimize Cost function
    # Grading Descent = First Value - Learning_Rate * ∂Cost_Function / ∂Parameter
    weights = []
    bias = []
    costs = []
    for i in range(iteration_num):
        weights.append(w[0])
        bias.append(b)
        y_head,cost = logistic_backward_propagation(x_train, y_train, w, b)
        costs.append(cost)
        
        w = w - learning_rate*((np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1])
        
        b = b - learning_rate*((np.sum(y_head-y_train))/x_train.shape[1])
    return weights,bias,costs,w,b
def logistic_backward_propagation(x_train, y_train, w, b):
    
    z = np.dot(w.T,x_train) + b
    
    y_head = sigmoid(z)     
    
    # To calculate how far our prediction from real value we use Log likehood Function
    # Log Likehood = (y * log(y_head)) + ((1−y) * log(1−y_head))
    # Difference between real values and predicted values is Cost Function our aim is to minimize Cost Function
    # Cost Function = ∑-(Log Likehood)
    
    inv_loss = (y_train * np.log(y_head)) + ((1-y_train) * np.log(1-y_head))
    
    cost = (-1/x_train.shape[1]) * (np.sum(inv_loss))
    
    return y_head,cost
def predict(w,b,x_test):
    
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate , iteration_num):
    # initialize
    dimension =  x_train.shape[0]
    
    w,b = weight_init(dimension)
    
    weights,bias,costs,w,b = grading_decent (w, b, x_train, learning_rate, iteration_num)
    
    y_prediction_test = predict(w, b, x_test)
    y_prediction_train = predict(w, b, x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
    return weights,costs
weights,costs = logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, iteration_num = 150)
plt.scatter(weights,costs)
plt.show()
weights,costs = logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, iteration_num = 300)
plt.scatter(weights,costs) 
plt.show()
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=150)
lr.fit(x_train.T,y_train.T)
lr.predict(x_test.T)
score = lr.score(x_test.T, y_test.T)
print(score)
x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
# Split the data

x_train,x_test,y_train,y_test = train_test_split(x_norm, y, test_size=0.2, random_state=42)
# Pandas Dataframe create arrays with features on column and observations on rows.
# We need to transpose these arrays to fit Logistic Regression Equation   
x_test = x_test.T
x_train = x_train.T
weights,costs = logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, iteration_num = 150)