# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#we will need these libraries 

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import fmin_tnc



from sklearn.datasets import make_classification

import seaborn as sns
#this fuction will give us a probablity value bases on the input, 



def sigmoid(x):

    return 1 / (1 + np.exp(-x))





# Computes the weighted sum of inputs, we can feed this into sigmoid function

# to get our probablility

    

def net_input(theta, x):

    return np.dot(x, theta)



#we can create a probablity function to put together the math for the 

#above methods, this function will give us a value between 0 and 1





def probability(theta, x):

    # Returns the probability after passing through sigmoid

    return sigmoid(net_input(theta, x))

#this fuction computes the current cost, bases on our current weights

#we need to recude this cost to get more accurate results,



def cost_function(theta, x, y):

    

    m = x.shape[0]

    total_cost = -(1 / m) * np.sum(

        y * np.log(probability(theta, x)) + (1 - y) * np.log(

            1 - probability(theta, x)))

    return total_cost
# Computes the gradient of the loss function at the point theta

# ie. it calculates the gradient as per our current weights



# we will use this gradient value to update our weights

def gradient(theta, x, y):

    m = x.shape[0]

    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)
from scipy import optimize



#fmin_tnc can be used to minimize the cost fuction 



# def fit(x, y, thetaa):

#     opt_weights = fmin_tnc(func=cost_function, x0=theta,

#                   fprime=gradient,args=(x, y.flatten()))

#     return opt_weights[0]



#fitting to gain optimal weights





def gradient_descent(X, y, beta, lr=0.1): 

    

    cost = cost_function(beta, X, y) 

    converge = 0.0001

    change_cost = 1.0

    num_iter = 0

      

    #we want to end training , ie calculating the weights when 

    #our change of cost goes from to as close or equal to our convergence

    #the lower our convergence the more iterations we will perform

    

    #here i have taken the convergence at 0.0001

    #ive set learning rate to 0.1 as this is giving me the best results

    

    while(change_cost>converge):

        old_cost = cost 

        

        #updating our weights with our gradient at current weights

        beta = beta - (lr * gradient(beta, X, y)) 

        

        #calculating the cost with updated weights

        cost = cost_function(beta, X, y) 

        

        #updating the change in cost to step towards the convergence

        

        change_cost = old_cost - cost

        

        #keeping track of  the number of iterations

        num_iter += 1

        

    #want to display number of iterations performed and final cost    

    print(num_iter)

    print(cost)          

    

    #returning the optimal weights

    return beta 
def normalize(X):   

    mins = np.min(X, axis = 0) 

    maxs = np.max(X, axis = 0) 

    rng = maxs - mins 

    norm_X = 1 - ((maxs - X)/rng) 

    return norm_X 
data = pd.read_csv("/kaggle/input/telecocustomerchurn/WA_Fn-UseC_-Telco-Customer-Churn.csv")





data["TotalCharges"] = data["TotalCharges"].apply(lambda n : 123 if n == " " else pd.to_numeric(n)).values

X = data[["tenure","MonthlyCharges","TotalCharges"]].values

y = data["Churn"].apply(lambda x : 1 if x == "Yes" else 0).values



# #load the data using pandas

# data = pd.read_csv("/kaggle/input/markss/marks.txt",names=["t1","t2","admin"])



# #X will be an array that will contain the test marks for both tests

# ##TRAINING FEATURS

# ## extracting first two columns of our dataset

# X = data.iloc[:, :-1]



# #y will be the array that will contain admitted/not admitted boolen value(1/0)

# #extracting last column

# ##TARGET VARIABLE

# y = data.iloc[:, -1]



# #displaying 

# X,y
#we need to normalize our training features to get a good result

X = normalize(X)





#concat a row of 1s to our training dataset

X = np.c_[np.ones((X.shape[0], 1)), X]



#change the shape of y so its a vertical array

y = y[:, np.newaxis]



#set our default/initialised weights at 0, we need to make sure that this vector

#has as many elements as training set 

theta = np.zeros((X.shape[1], 1))



#displaying 

X[0:5],y[0:5],theta
parameters=gradient_descent(X, y,theta)
#we will use the probablity function again to get probablity values for our

#data with our optimal weights.



#these are our predicted values 



def predict(x,theta):

    #theta = parameters#[:, np.newaxis]

    return probability(theta, x)



#this method will calculate accuracy

def accuracy(x, actual_classes):

    

    #use predict method to make predictions for our data,with 

    #optimised parameters

    

    #if the predicted probablity is more than or eual to 0 consider it 1

    #if not 0

    predicted_classes = (predict(x,parameters) >= 0.5).astype(int)

    

    #flattening this list of prediction

    predicted_classes = predicted_classes.flatten()

    

    #we can take the mean of whenever our prediction match the existing data

    accuracy = np.mean(predicted_classes == actual_classes)

    

    #if we multiply this mean by 100 we get accuracy percentage

    return accuracy * 100
accuracy(X, y.flatten())