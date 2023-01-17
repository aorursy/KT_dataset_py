import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



np.random.seed(42)

X = 2 * np.random.rand(100,1)

Y = 10 + 4 * X+np.random.randn(100,1) 



plt.style.use('seaborn')

plt.scatter(X, Y, color='black')

plt.title('Training Data')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()
n = float(len(X)) # number of data pair

init_b = 0

init_m = 0



def compute_cost_function_point_by_point(m, b, X, Y):

    totalError = 0

    

    for i in range(len(X)):

        x = X[i,0]

        y = Y[i,0]

        totalError += (y - (m * x + b)) ** 2 

        

    return totalError / n



compute_cost_function_point_by_point(init_m,init_b,X,Y)
learning_rate = 0.01
def compute_gradient_step_by_step(m,b,X,Y,learning_rate):

    b_gradient = 0

    m_gradient = 0

    for i in range(len(X)):

        x = X[i,0]

        y = Y[i,0]

        

        b_gradient += -(2/n) * (y - ((m * x) + b))

        m_gradient += -(2/n) * x * (y - ((m * x) + b))

    new_b = b - (learning_rate * b_gradient)

    new_m = m - (learning_rate * m_gradient)

    return [new_b,new_m]
num_iteration = 1000

min_step_size = 0.001
def gradient_descent(X,Y,starting_b,starting_m,learning_rate, num_iteration):

    b = starting_b

    m = starting_m

    i = 0

    while i in range(num_iteration) or (b < min_step_size and m < min_step_size):

        b, m = compute_gradient_step_by_step(m, b, X, Y, learning_rate)

        i += 1

    return [b,m]
def calculate_predicted_values(X, opt_b, opt_m):

    return X*opt_m + opt_b
[g_b, g_m] = gradient_descent(X,Y,init_b,init_m,learning_rate, num_iteration)
opt_X = calculate_predicted_values(X, g_b, g_m)
plt.style.use('seaborn')

plt.scatter(X, Y, color='black')

plt.plot(X, opt_X)

plt.title('Linear Regression with Gradient Descent ')

plt.xlabel('Input')

plt.ylabel('Output')

plt.show()