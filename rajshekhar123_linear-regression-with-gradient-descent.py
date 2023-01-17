# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



def Linear_Regression(x,y):

    #mean of x and y

    mean_x,mean_y = np.mean(x),np.mean(y)

    n = np.size(x)



    #calculating the slope and sum of square error

    ss_xy = n*(np.sum(x)*np.sum(y)) - np.sum(x)*np.sum(y) 

    ss_xx = n*(np.sum(x))**2 - np.sum(x)**2 



    # Y = Bo + B1*X

    B1 = ss_xy/ss_xx #calculating the slope of the regression line/weight

    Bo = mean_y - B1*mean_x #calculating the intercept/bias of the line

    return (B1,Bo)



def plot_regression(x,y,b):

    plt.scatter(x,y,color='red',marker='+')

    y_pred = b[0]*x + b[1]

    plt.plot(x,y_pred,color='green')

    plt.title('Trained model ')

    plt.show()

    return y_pred   



def Gradient_descent(x,y,b,y_predicted):

    print('Parammeters using Gradient Descent')

    m_curr = b[0]

    b_curr = b[1]

    rate = 0.01

    n = len(x)

    plt.scatter(x,y,color='red',marker='+')

    for i in range(100):

        y_predicted = m_curr * x + b_curr

        cost = (1/n)*sum([val**2 for val in(y-y_predicted)])    ### THIS IS WITH GRADIENT DESCENT

        plt.plot(x,y_predicted,color='orange')

        md = -(2/n)*sum(x*(y-y_predicted))

        yd = -(2/n)*sum(y-y_predicted)

        m_curr = m_curr - rate * md

        b_curr = b_curr - rate * yd

        print('m={},b={},cost={}'.format(m_curr,b_curr,cost))

    plt.plot(x,y_predicted,color='blue')     



def main() :

    #reading the data

    x = np.array([1,2,3,4,5])

    y = np.array([5,7,9,11,13])

    b = Linear_Regression(x,y)

    print("The estimated coefficents of line:- \n B1 = {} \n Bo = {}".format(b[0],b[1]))

    train_data = plot_regression(x,y,b)

    Gradient_descent(x,y,b,train_data)



main()

# Any results you write to the current directory are saved as output.