# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from wordcloud import WordCloud

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataframe = pd.read_csv('../input/voice.csv') #read to file
newDataframe = dataframe.loc[:,["label"]]
import missingno as msno
msno.matrix(newDataframe)
plt.show()
msno.bar(newDataframe)
plt.show()
dataframe.head() #TOP10 data
dataframe.label = [1 if(each == 'male') else 0 for each in dataframe.label] #1:Male 0: Female
y = dataframe.label.values
x_data = dataframe.drop(["label"],axis=1) 
x = (x_data - np.min(x_data))/(np.max(x_data) - np.min(x_data)).values #Normalization

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test  = x_test.T
y_train = y_train.T
y_test  = y_test.T
def init_weights_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b
def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z))
    return y_head
def forward_backward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]    
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] 
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost, gradients
def update_weights_bias(w,b,x_train,y_train,learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    for i in range(number_of_iterarion):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
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
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    dimension = x_train.shape[0]
    w, b = init_weights_bias(dimension)

    parameters,gradients,cost_list = update_weights_bias(w,b,x_train,y_train,learning_rate,num_iterations)

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return y_prediction_test #Estimates for Complex Matrix
    
y_predict = logistic_regression(x_train, y_train, x_test, y_test, learning_rate=1, num_iterations=300)
predict = []
for i in range(0,1):
    for each in y_predict[i]:
        predict.append(int(each))
truePredict = 0
falsePredict = 0
for p in range(len(predict)):
    for y in range(p,len(y_test)):
        if (predict[p] == y_test[y]):
            truePredict = truePredict +1
            break
        else:
            falsePredict = falsePredict +1
            break
print("True Predict: ",truePredict)
print("False Predict",falsePredict)
print("-------------------------------------------------------------------------------------")
print("Predict: ",predict)
print("-------------------------------------------------------------------------------------")
print("y_test: ",y_test)
x_Axis = ["True","False"]
y_Axis = [truePredict,falsePredict]

plt.figure(figsize=(15,15))
sns.barplot(x=x_Axis,y=y_Axis,palette = sns.cubehelix_palette(len(x_Axis)))
plt.xlabel("Gender Class")
plt.ylabel("Frequency")
plt.title("Male and Female")
plt.show()
conf_matrix = confusion_matrix(y_test,predict)
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(conf_matrix,annot=True,linewidths=0.5,linecolor="white",fmt=".0f",ax=ax)
plt.xlabel("y_test")
plt.ylabel("predict")
plt.show()