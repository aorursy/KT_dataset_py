import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from numpy import *



# y = mx + b

# m is slope, b is y-intercept

def compute_error_for_line_given_points(b, m, points):

    totalError = 0

    for i in range(0, len(points)):

        x = points[i, 0]

        y = points[i, 1]

        totalError += (y - (m * x + b)) ** 2

    return totalError / float(len(points))



def step_gradient(b_current, m_current, points, learningRate):

    b_gradient = 0

    m_gradient = 0

    N = float(len(points))

    for i in range(0, len(points)):

        x = points[i, 0]

        y = points[i, 1]

        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))

        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learningRate * b_gradient)

    new_m = m_current - (learningRate * m_gradient)

    return [new_b, new_m]



def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):

    b = starting_b

    m = starting_m

    for i in range(num_iterations):

        b, m = step_gradient(b, m, array(points), learning_rate)

    return [b, m]



def run():

    points = genfromtxt("../input/linear_demo.csv", delimiter=",")

    learning_rate = 0.0001

    initial_b = 0 # initial y-intercept guess

    initial_m = 0 # initial slope guess

    num_iterations = 1000

    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))

    print ("Running...")

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    #print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    print (f' after iteration={num_iterations}  b={b} m={m}', "error=",compute_error_for_line_given_points(b, m, points)) 

    

    x,y = genfromtxt("../input/linear_demo.csv",unpack=True, delimiter=",")

    plt.scatter(x,y)

    new_y=m*x+b

    plt.plot(x, new_y, '-b')

    plt.show()

if __name__ == '__main__':

    run()