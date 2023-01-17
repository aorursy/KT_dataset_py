# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from skimage.transform import resize



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data.isnull().sum()
data.info()

data.describe()
data.head(5)

data.tail(5)
y = data.target.values

x_data = data.drop(["target"], axis=1)
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

x = min_max_scaler.fit_transform(x_data)

print(x)
f,ax = plt.subplots(figsize=(12,12))

sns.heatmap(data.corr(), annot=False, linewidths=.5, fmt =".1f=", ax=ax)

plt.show()
import warnings

warnings.filterwarnings('ignore')



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (10, 3)

sns.distplot(data['age'], color = 'blue')

plt.title('Range of Age', fontsize = 17)

plt.show()




size = data['sex'].value_counts()

colors = ['blue', 'green']

labels = "Male", "Female"

explode = [0, 0]



my_circle = plt.Circle((0, 0), 0.4, color = 'white')



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')

plt.title('Gender', fontsize = 17)

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.legend()

plt.show()


x_thalach = data.thalach.values.reshape(-1,1)

y = y.reshape(-1,1)



from sklearn.linear_model import LinearRegression

linreg = LinearRegression()

linreg.fit(x_thalach,y)



y_headlinreg = linreg.predict(x_thalach)



plt.scatter(x_thalach,y)

plt.xlabel("thalach")

plt.ylabel("target")



plt.plot(x_thalach,y_headlinreg, color= "green")

plt.show()



from sklearn.metrics import r2_score



print("r_square score: ", r2_score(y,y_headlinreg))







from sklearn.preprocessing import PolynomialFeatures

polynomial_regression = PolynomialFeatures(degree = 4)



x_polynomial = polynomial_regression.fit_transform(x)



linear_regression2 = LinearRegression()

linear_regression2.fit(x_polynomial,y)

y_head2 = linear_regression2.predict(x_polynomial)



print("r_square score: ", r2_score(y,y_head2))

y= y.ravel() #the returned array will have the same type as the input array

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state= 42)

print(x_train)

x_train = x_train.T

x_test = x_test.T

y_train = y_train.T

y_test = y_test.T

print(x_train)



def initialize_weights_and_bias (dimension):

    w= np.full((dimension,1),0.01)

    b = 0.0 

    return w,b





def sigmoid(z) :

    y_head = 1/(1+np.exp(-z))

    return y_head



def forward_backward_propagation (w,b,x_train,y_train):

    z = np.dot(w.T,x_train)+b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]

    

    

    #backward propogation 

    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients



#updating learning parameters

def update(w,b,x_train,y_train,learning_rate,number_of_iterations):

    cost_list=[]

    cost_list2=[]

    index=[]

    

#updating learning parameters for number of iteration times

    for i in range (number_of_iterations):

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

    

        w= w-learning_rate*gradients[derivative_weight]

        b= b-learning_rate*gradients[derivative_bias]

        if i %10 ==0 :

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

        

    

        parameters = {"weights" : w , "bias" : b}

        plt.plot(index,cost_list2)

        plt.xticks(index,rotation="vertical")

        plt.xlabel("Number of iteration")

        plt.ylabel("cost")

        plt.show()

        return parameters, gradients, cost_list

    

#predict

def predict(w,b,x_test):

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1

                

            return Y_prediction

    

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  number_of_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,number_of_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    

    logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train.T,y_train.T)

print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)



# knn model

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)

print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

 

# %% test

print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))