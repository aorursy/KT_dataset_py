from numpy import *
import matplotlib.pyplot as plt
def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')

def mean_squared_error(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))
def batch_gradient(b_current, m_current, points, learningRate):
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

def gradient_descent_runner(points, starting_b, starting_m, learning_rate):
    b = starting_b
    m = starting_m
    checker = True
    while(checker):
        b_pre, m_pre = b,m
        error_bef_grad = mean_squared_error(b, m, points)
        b, m = batch_gradient(b, m, array(points), learning_rate)
        error_aft_grad = mean_squared_error(b, m, points)
        if error_aft_grad > error_bef_grad:
            checker = False
    plot_best_fit(b_pre,m_pre)
    return[b_pre,m_pre]
def main_fun():
    points = genfromtxt("../input/distancecycledvscaloriesburned/data.csv", delimiter=",")
    plt.plot(points[:,0], points[:,1], '.')
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, mean_squared_error(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate)
    print("After Gradient descent b = {0}, m = {1}, error = {2}".format(b, m, mean_squared_error(b, m, points)))
main_fun()