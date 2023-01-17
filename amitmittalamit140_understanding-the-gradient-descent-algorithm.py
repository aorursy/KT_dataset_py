#importing libraries which will be required in the process

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#supress warnings

import warnings

warnings.filterwarnings("ignore")
#loading dataset using pandas and showing first five rows

df = pd.read_csv('/kaggle/input/advertising-data/Advertising.csv')

df.head()
# X -> Independent variable, which is TV

X = df.iloc[:,1:2].values



# y -> Dependent varible, which is Sales

y = df.iloc[:,-1:].values
#Now we will see a scatter plot to show relation between independent and dependent varaiable

plt.scatter(X,y,alpha=0.5)

plt.xlabel('TV',size=10)

plt.ylabel('Sales',size=10)

plt.title('Scatter Plot of TV vs Sales',size=15)

plt.show()
def FeatureScaling(X):

    """

    is function takes an array as an input, which needs to be scaled down.

    Apply Standardization technique to it and scale down the features with mean = 0 and standard deviation = 1

    

    Input <- 2 dimensional numpy array

    Returns -> Numpy array after applying Feature Scaling

    """

    mean = np.mean(X,axis=0)

    std = np.std(X,axis=0)

    for i in range(X.shape[1]):

        X[:,i] = (X[:,i]-mean[i])/std[i]



    return X
# scaling variable TV using defined function

X = FeatureScaling(X)
print("X before adding column of 1's:",X[:2],sep="\n")

X = np.append(arr=np.ones((X.shape[0],1)),values=X,axis=1)

print("\nX after adding column of 1's:",X[:2],sep="\n")
#ComputeCost function determines the cost (sum of squared errors) 



def ComputeCost(X,y,theta):

    """

    This function takes three inputs and uses the Cost Function to determine the cost (basically error of prediction vs

    actual values)

    Cost Function: Sum of square of error in predicted values divided by number of data points in the set

    J = 1/(2*m) *  Summation(Square(Predicted values - Actual values))

    

    Input <- Take three numoy array X,y and theta

    Return -> The cost calculated from the Cost Function

    """

    m=X.shape[0] #number of data points in the set

    J = (1/(2*m)) * np.sum((X.dot(theta) - y)**2)

    return J
#Gradient Descent Algorithm to minimize the Cost and find best parameters in order to get best line for our dataset



def GradientDescent_New(X,y,theta,alpha,no_of_iters):

    """

    Gradient Descent Algorithm to minimize the Cost

    

    Input <- X, y and theta are numpy arrays

            X -> Independent Variables/ Features

            y -> Dependent/ Target Variable

            theta -> Parameters 

            alpha -> Learning Rate i.e. size of each steps we take

            no_of_iters -> Number of iterations we want to perform

        Return -> theta (numpy array) which are the best parameters for our dataset to fit a linear line

             and Cost Computed (numpy array) for each iteration

    """

    m=X.shape[0]

    J_Cost = []

    theta_array = []

    for i in range(no_of_iters):

        error = np.dot(X.transpose(),(X.dot(theta)-y))

        theta = theta - alpha * (1/m) * error

        J_Cost.append(ComputeCost(X,y,theta))

        

        #below code is to note theta value of every 30th iteration, which we will be using further in this notebook

        if (i+1)%30 == 0:

            theta_array.append(theta)

    

    return theta, np.array(J_Cost), np.array(theta_array)
#number of iterations

iters = 300



#learning rate

alpha = 0.01



#initializing theta

theta = np.zeros((X.shape[1],1))



#finally computing values using function

theta, J_Costs, theta_array = GradientDescent_New(X,y,theta,alpha,iters)
plt.figure(figsize=(8,5))

plt.plot(J_Costs,color="g")

plt.title('Convergence of Gradient Descent Algorithm')

plt.xlabel('No. of iterations')

plt.ylabel('Cost')

plt.show()
# Removing column of 1's from X in order to visualize the data.

X = X[:,1:]
for i in range(10):

    plt.figure(figsize=(40,10))

    plt.subplot(2,5,i+1)

    b0, b1 = round(float(theta_array[i,0]),2), round(float(theta_array[i,1]),2)

    y_pred = b0 + b1 * X

    mse = round(J_Costs[30*i+30-1],2)

    plt.scatter(X,y,alpha=0.5)

    plt.plot(X,y_pred,color="r")

    plt.xlabel('TV',size=10)

    plt.ylabel('Sales',size=10)

    plt.title('Sales = {} + {} * TV (after {} iterations, MSE: {})'.format(b0,b1,30*i+30,mse),size=14)

    plt.show()