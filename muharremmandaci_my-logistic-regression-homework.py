# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/weatherAUS.csv")
df.head()
df.info()
wind_directions_list = df.WindGustDir
locations_list = df.Location
df.drop(["Date", "Location", "WindGustDir", "WindDir9am", "WindDir3pm"], axis=1, inplace=True)
df.head()
df.RainToday = [1 if each=="Yes" else 0 for each in df.RainToday]
df.RainTomorrow = [1 if each=="Yes" else 0 for each in df.RainTomorrow]
df.head()
# completing missing datas
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

missingData = df.iloc[:,0:].values
imputer = imputer.fit(missingData)               # missingDatas[:,0:3]
completedData = imputer.transform(missingData)   # missingDatas[:,0:3]

df.iloc[:, 0:] = completedData
df.head(10)
wind_directions_list = wind_directions_list.fillna("UNKNOWN")
wind_directions_list.tail(10)
location_names=[]
for each in locations_list:
    if not each in location_names:
        location_names.append(each)
location_names.sort()

WindGustDir_names=[]
for each in wind_directions_list:
    if not each in WindGustDir_names:
        WindGustDir_names.append(each)
WindGustDir_names.sort()
WindGustDir = wind_directions_list.values.reshape(-1,1)
locations = locations_list.values.reshape(-1,1)

labelEncoder = preprocessing.LabelEncoder()

locations[:, 0] = labelEncoder.fit_transform(locations[:, 0])
WindGustDir[:, 0] = labelEncoder.fit_transform(WindGustDir[:, 0])

oneHotEncoder = preprocessing.OneHotEncoder(categorical_features='all')

locations = oneHotEncoder.fit_transform(locations).toarray()
WindGustDir = oneHotEncoder.fit_transform(WindGustDir).toarray()

dfLocations = pd.DataFrame(data=locations, index=range(145460), columns=location_names)
dfWindGustDir = pd.DataFrame(data=WindGustDir, index=range(145460), columns=WindGustDir_names)

df_with_wind_dir = pd.concat([dfLocations, df, dfWindGustDir], axis=1)
df_without_wind_dir = pd.concat([dfLocations, df], axis=1)

print(df_with_wind_dir.tail())
print(df_without_wind_dir.tail())
y_no_dir = df_without_wind_dir.RainTomorrow.values.reshape(-1,1)
x_data_no_dir = df_without_wind_dir.drop(["RainTomorrow"], axis=1)
x_no_dir = (x_data_no_dir - np.min(x_data_no_dir))/(np.max(x_data_no_dir)-np.min(x_data_no_dir)).values

y_dir = df_with_wind_dir.RainTomorrow.values.reshape(-1,1)
x_data_dir = df_with_wind_dir.drop(["RainTomorrow"], axis=1)
x_dir = (x_data_dir - np.min(x_data_dir))/(np.max(x_data_dir)-np.min(x_data_dir)).values
from sklearn.model_selection import train_test_split
x_train_no_dir, x_test_no_dir, y_train_no_dir, y_test_no_dir = train_test_split(x_no_dir, y_no_dir, test_size = 0.2, random_state=42)
x_train_dir, x_test_dir, y_train_dir, y_test_dir = train_test_split(x_dir, y_dir, test_size = 0.2, random_state=42)
def initialize_weights_and_bias(dimension):
    w=np.full((dimension, 1), 0.01)
    b=0.0
    return w,b


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


def forward_backward_propagation(w, b, x_train, y_train):
    # forward propagation
    z=np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_head)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    # backward propagation
    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost, gradients


def update(w, b, x_train, y_train, learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


 # prediction
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


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension = x_train.shape[0]  
    w, b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    return 100 - np.mean(np.abs(y_prediction_test - y_test)) * 100    
acc_dir = logistic_regression(x_train_dir.T, y_train_dir.T, x_test_dir.T, y_test_dir.T,learning_rate = 2, num_iterations = 300)
acc_no_dir = logistic_regression(x_train_no_dir.T, y_train_no_dir.T, x_test_no_dir.T, y_test_no_dir.T,learning_rate = 2, num_iterations = 300)

print("accuracy without wind direction: % {:6.3f}".format(acc_no_dir))
print("accuracy with wind direction: % {:6.3f}".format(acc_dir))
from sklearn.linear_model import LogisticRegression

lr_dir=LogisticRegression()
lr_dir.fit(x_train_dir, y_train_dir)

lr_no_dir=LogisticRegression()
lr_no_dir.fit(x_train_no_dir, y_train_no_dir)

acc_no_dir = lr_no_dir.score(x_test_no_dir, y_test_no_dir)*100
print("test accuracy without wind direction: % {:6.3f}".format(acc_no_dir))

acc_dir = lr_dir.score(x_test_dir, y_test_dir)*100
print("test accuracy with wind direction: % {:6.3f}".format(acc_dir))