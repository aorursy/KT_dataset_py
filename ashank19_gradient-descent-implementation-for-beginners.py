# Importing the dataset after uploading the dataset



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df=pd.read_csv("../input/real_estate_price_size.csv")



x = df['size(X)']

y = df['price(Y)']



x = (x - x.mean()) / x.std()

x = np.c_[np.ones(x.shape[0]), x] 
alpha = 0.05 #Step size

iterations = 100 #No. of iterations

m = y.size #No. of data points

np.random.seed(0) #Set the seed

theta = np.random.rand(2) #Pick some random values to start with
# Gradient descent algorithm

def gradientdescent(x,y,theta,num_iter,alpha):

    j_history=[]

    thetas=[theta]

    for i in range(num_iter):

        error=np.dot(x,theta)-y

        # Updating theta after every iteration

        theta=theta-((alpha/m)*np.dot(x.T,error))

        thetas.append(theta)

        j_history.append(cost(x,y,theta))

    return j_history,thetas
# Calculating cost function for every value of updated theta

def cost(x,y,theta):

    j=1/(2*m)*sum((np.dot(x,theta)-y)**2)

    return j
#Storing the cost function and values of theta after every iteration in a list

cost_function ,past_thetas=gradientdescent(x,y,theta,iterations,alpha)