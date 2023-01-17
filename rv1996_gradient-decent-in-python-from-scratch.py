#standard import

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

# To visualise in the notebook

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/advertising.csv')
#TV Radio NewsPapers are independent variable and Sales in dependent variable

data.head()
#exploring the data - (no need to clean it as it was already cleaned)

data.info()
#the data variable must be normalized

data_nor = (data - data.mean())/data.std()

data_nor
sns.pairplot(data_nor)
# we can run gradient decent on two variables like sales and tv

X = np.array(data_nor['TV'])

Y = np.array(data_nor['Sales'])
#ploting the data for only X and Y we get

sns.pairplot(data_nor, x_vars='TV', y_vars='Sales',height=5, kind='scatter')
# implementing the gradient decent we get

# we'll start the initial value of m(Slope) and c(Intercept) to be as 0

# iters it the number of iteration and rate is the learing rate (because) we are using the iterative method not closed

# for minimization





def gradient(X, y, m_curr=0, c_curr=0, iters=1000, rate=0.01):

    N = float(len(y)) # length of the data set

    gd_df = pd.DataFrame(columns = ['m_curr', 'c_curr','cost'])

    for i in range(iters):

        y_curr = (m_curr * X) + c_curr # or model with the current slope and intercept

        cost =sum([data**2 for data in (y-y_curr)]) / N # (y-y_current) diff square

        m_grad = -(2/N) * sum(X * (y - y_curr))

        c_grad = -(2/N) * sum(y - y_curr)

        m_curr = m_curr - (rate * m_grad)

        c_curr = c_curr - (rate * c_grad)

        gd_df.loc[i] = [m_curr,c_curr,cost]

    return gd_df



    
cost = gradient(X=X,y=Y)

cost
cost.reset_index().plot.line(x='index',y=['cost'],figsize=(8,6))
#above we have only consider the variable for tv and sales let use more variable now

X_multi = data_nor[['TV','Radio','Newspaper']]

Y = data_nor['Sales']
# adding and intercept column to X

X_multi['Intercept'] = 1



X_multi = X_multi.reindex(['Intercept','TV','Radio','Newspaper'],axis=1)

X_multi
#converting X and Y to numpy arrays

X_multi = np.array(X_multi)

Y = np.array(Y)
# we need as vector representation for the intercept i.e intercept c, tv, newspaper, radio 

# these are all the independent variable on which the dependent variable sales depend upon

theta = np.matrix(np.array([0,0,0,0]))

alpha = 0.01 # learing rate

iteration = 1000

#now for defining our cost function we have use theta to get the cost

#this will be used in simultanious theta updates



def compute_cost(X, y, theta):

    return np.sum(np.square(np.matmul(X, theta) - y))/(2*len(y))
#gradient decent for multiple variable would be



def gradient_multi(X,y,theta,alpha,iteration):

    theta = np.zeros(X.shape[1])

    m = len(X)

    gdm_df = pd.DataFrame( columns = ['Bets','cost'])

    

    for i in range(iteration):

        gradient = (1/m) * np.matmul(X.T, np.matmul(X, theta) - y)

        theta = theta - alpha * gradient

        cost = compute_cost(X, y, theta)

        gdm_df.loc[i] = [theta,cost]



    return gdm_df

    
cost = gradient_multi(X_multi, Y, theta, alpha, iteration)
cost.reset_index().plot.line(x="index",y="cost",figsize=(8,6))
# event for the multiple variable the gradient decent is acting as a cost optimization function
cost
# import LinearRegression from sklearn

from sklearn.linear_model import LinearRegression



# Representing LinearRegression as lr(Creating LinearRegression Object)

lr = LinearRegression()



#You don't need to specify an object to save the result because 'lr' will take the results of the fitted model.

lr.fit(X_multi, Y)
#Calculated vaue using sklearn

print(lr.intercept_)

print(lr.coef_)
print(cost.tail(1)['Bets'])
# slight difference in coefficient because of the change in algorithm used by sklearn