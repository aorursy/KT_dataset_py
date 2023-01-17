# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import random

import sklearn.datasets

from sklearn.datasets.samples_generator import make_regression 

import pylab

from scipy import stats

import matplotlib.pyplot as plt




def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):

    converged = False

    iter = 0

    m = x.shape[0] # number of samples



    # initial theta

    t0 = np.random.random(x.shape[1])

    t1 = np.random.random(x.shape[1])



    # total error, J(theta)

    J = sum([(t0 + t1*x[i] - y[i])**2 for i in range(m)])



    # Iterate Loop

    while not converged:

        # for each training sample, compute the gradient (d/d_theta j(theta))

        grad0 = 1.0/m * sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 

        grad1 = 1.0/m * sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])



        # update the theta_temp

        temp0 = t0 - alpha * grad0

        temp1 = t1 - alpha * grad1

    

        # update theta

        t0 = temp0

        t1 = temp1



        # mean squared error

        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) 



        if abs(J-e) <= ep:

            print ('Converged, iterations: ', iter)

            converged = True

    

        J = e   # update error 

        iter += 1  # update iter

    

        if iter == max_iter:

            print('Max interactions exceeded!')

            converged = True



    return t0,t1



if __name__ == '__main__':



    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 

                        random_state=0, noise=35) 

    print ('x.shape = {} y.shape = {}'.format(x.shape, y.shape))



    plt.plot(x,y,'o')

    plt.xlim([-2,2.5])

    plt.ylim([-150,150])

    plt.show()


    alpha = 0.01 # learning rate

    ep = 0.01 # convergence criteria



    # call gredient decent, and get intercept(=theta0) and slope(=theta1)

    theta0, theta1 = gradient_descent(alpha, x, y, ep, max_iter=1000)

    print ('theta0 = {} theta1 = {}'.format(theta0, theta1))



    # check with scipy linear regression 

    slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)

    print ('intercept ={} slope ={}'.format(intercept, slope))
    # plot

    for i in range(x.shape[0]):

        y_predict = theta0 + theta1*x 



    pylab.plot(x,y,'o')

    pylab.plot(x,y_predict,'k-')

    pylab.show()