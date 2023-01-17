# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def errorForGivenPoints(b,m,points):

    totalErrors = 0

    for i in range(len(points)):

        x = points[i,0]

        y = points[i,1]

        totalErrors += (y- (m * x) + b) ** 2

    return(totalErrors/float(len(points)))
def step_gradient(b_current, m_current, points, learning_rate):

    b_gradient = 0

    m_gradient = 0

    N = float(len(points))

    for i in range(len(points)):

        x = points[i,0]

        y = points[i,1]

        b_gradient += (-2/N) * (y - ((m_current * x) + b_current))

        m_gradient += (-2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)

    new_m = m_current - (learning_rate * m_gradient)

    return([new_b,new_m])
def gradient_descent(points,initial_b,initial_m,learning_rate,num_iterations):

    b = initial_b

    m = initial_m

    for i in range(num_iterations):

        b,m = step_gradient(b, m, np.array(points), learning_rate)

#         print(errorForGivenPoints(b,m,points))

#         print("value of {0} and {1}".format(b,m))

    return [b,m]
def run():

    points = np.genfromtxt('../input/data.csv',delimiter=',')

    learning_rate = 0.0001 # hyper-parameter

#      y = mx + b

    initial_b = 0

    initial_m = 0

    num_iterations = 1000

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b,initial_m,errorForGivenPoints(initial_b,initial_m,points)))

    print("Working...")

    [b,m] = gradient_descent(points,initial_b,initial_m,learning_rate,num_iterations)

    print("After {0} iterations b = {1}, m = {2}, error= {3}".format(num_iterations,b,m,errorForGivenPoints(b,m,points)))
if __name__ == '__main__':

    run()