

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data_voice = pd.read_csv("/kaggle/input/voicegender/voice.csv")



data = data_voice.copy()



data.tail()
data.describe().T
data.corr()
data.info()
data.isnull().sum()
msno.matrix(data)

plt.show()
plt.subplots(4,5,figsize=(15,15))

for i in range(1,21):

    plt.subplot(4,5,i)

    plt.title(data.columns[i-1])

    sns.kdeplot(data.loc[data['label'] == "female", data.columns[i-1]], color= 'red', label='Female')

    sns.kdeplot(data.loc[data['label'] == "male", data.columns[i-1]], color= 'blue', label='Male')
# import data again to avoid confusion



import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

data_voice = pd.read_csv("/kaggle/input/voicegender/voice.csv")

data = data_voice.copy()



male = data[data.label == "male"]



female = data[data.label == "female"]



# trace1

trace1 = go.Scatter3d(

    x=male.meanfun,

    y=male.IQR,

    z=male.Q25,

    mode='markers',

    name = "MALE",

    marker=dict(

        color='rgb(54, 170, 127)',

        size=12,

        line=dict(

            color='rgb(204, 204, 204)',

            width=0.1

        )

    )

)



trace2 = go.Scatter3d(

    x=female.meanfun,

    y=female.IQR,

    z=female.Q25,

    mode='markers',

    name = "FEMALE",

    marker=dict(

        color='rgb(217, 100, 100)',

        size=12,

        line=dict(

            color='rgb(255, 255, 255)',

            width=0.1

        )

    )

)



data1 = [trace1, trace2]

layout = go.Layout(

    title = ' 3D VOICE DATA ',

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    )

)

fig = go.Figure(data=data1, layout=layout)



iplot(fig)
#Normalization



data.label = [1 if each == "male" else 0 for each in data_voice.label]

y = data.label.values

x_data = data.drop(["label"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)



x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T


# w,b = Initialize Weights and Bias



def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b





# Sigmoid Function



def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head


def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) # Loss Function

    cost = (np.sum(loss))/x_train.shape[1]                      # Cost Function

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

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

        if i % 1000 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

            

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.grid(True)

    plt.title("We Update(learn) Parameters Weights and Bias")

    plt.show()

    return parameters, gradients, cost_list
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

    dimension =  x_train.shape[0]  

    w,b = initialize_weights_and_bias(dimension)

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.2, num_iterations = 15000)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

xx = print("train accuracy {} %".format(lr.score(x_train.T,y_train.T)*100))

yy = print("test accuracy {} %".format(lr.score(x_test.T,y_test.T)*100))


