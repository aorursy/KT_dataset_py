# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Create Salary Dataset



yr_x = [2,3,5,13,8,16,11,1,9]

salary_y = [15, 28,42,64, 50,90,58,8,54]

salary_df = pd.DataFrame({"yr_Exp":yr_x, "salary":salary_y})
salary_df["(x-xAvg)"] = salary_df["yr_Exp"] - np.mean(salary_df["yr_Exp"])

salary_df["(y-yAvg)"] = salary_df["salary"] - np.mean(salary_df["salary"])

salary_df["(x-xAvg)(y-yAvg)"] = salary_df["(x-xAvg)"]*salary_df["(y-yAvg)"]

salary_df["(x-xAvg)^2"] = salary_df["(x-xAvg)"]*salary_df["(x-xAvg)"]
salary_df.head()
# Plot Salary vs Years



fig, ax = plt.subplots()

ax.scatter(salary_df["yr_Exp"], salary_df["salary"], c="b")

plt.title("Salary Dataset", fontsize=16)

plt.xlabel("Years", fontsize=14)

plt.ylabel("Salary 10,000s", fontsize=14)

plt.show()
beta_1 = np.sum(salary_df["(x-xAvg)(y-yAvg)"])/np.sum(salary_df["(x-xAvg)^2"])

print("The slope of the line is : ",beta_1)
beta_0 = np.mean(salary_df["salary"]) - beta_1*np.mean(salary_df["yr_Exp"])

print("The intercept of the line is : ",beta_0)

print("\nSo the Linear Equation is y = {0:0.2f}x + {1:0.2f}".format(beta_1,beta_0))
# Some dummy data

x = salary_df["yr_Exp"]

y = salary_df["salary"]



# Find the slope and intercept of the best fit line



# Create a list of values in the best fit line

ols_predictions = [beta_1 * i + beta_0 for i in x]



# Plot the best fit line over the actual values

plt.scatter(x,y)

plt.plot(x, ols_predictions, 'r')

plt.title("Salary Dataset", fontsize=16)

plt.xlabel("Years", fontsize=14)

plt.ylabel("Salary 10,000s", fontsize=14)

plt.show()
# Create Normal Equation Function

def NormalEquation(X,Y):

    beta = np.linalg.inv(X.T @ X) @ (X.T @ Y)

    return beta
X = np.array([np.ones(len(salary_df["yr_Exp"])), salary_df["yr_Exp"]]).T

Y = salary_df["salary"][:, np.newaxis]
X
Y
NormalEquation(X,Y)
# y = b1x + b0

# b1 is slope, b0 is y-intercept



def compute_cost_for_line_given_points(b0, b1, points):

    totalError = 0

    

    for i in range(0, len(points)):

        x = points[i, 0]

        y = points[i, 1]

        totalError += (y - (b1 * x + b0)) ** 2

    return totalError / (2*float(len(points)))
def step_gradient(b0_current, b1_current, points, learningRate):

    b0_gradient = 0

    b1_gradient = 0

    N = float(len(points))



    for i in range(0, len(points)):

        x = points[i, 0]

        y = points[i, 1]

        b0_gradient += -(2/N) * (y - ((b1_current * x) + b0_current))

        b1_gradient += -(2/N) * x * (y - ((b1_current * x) + b0_current))

    new_b0 = b0_current - (learningRate * b0_gradient)

    new_b1 = b1_current - (learningRate * b1_gradient)

    return [new_b0, new_b1]
def gradient_descent(points, starting_b0, starting_b1, learning_rate, num_iterations):

    b0 = starting_b0

    b1 = starting_b1

    cost_history = np.zeros(num_iterations)  # create a vector to store the cost history

    for i in range(num_iterations):

        cost_history[i] = compute_cost_for_line_given_points(b0, b1, points) # compute and record the cost

        b0, b1 = step_gradient(b0, b1, points, learning_rate)



    return [b0, b1], cost_history
salary_df.head()
points = np.ones(shape=(len(salary_df["yr_Exp"]), 2))



points[:, 0] = salary_df["yr_Exp"]

points[:, 1] = salary_df["salary"]
points.shape
learning_rate = 0.008

initial_b0 = np.random.randn(1,1)*0.01

initial_b1 = np.random.randn(1,1)*0.01

num_iterations = 2000



print("Starting gradient descent at b0 = {0}, b1 = {1}, error = {2}".format(initial_b0, initial_b1, compute_cost_for_line_given_points(initial_b0, initial_b1, points)))

print("Running...")
[b0, b1], _ = gradient_descent(points, initial_b0, initial_b1, learning_rate, num_iterations)



print("After {0} iterations b0 = {1}, b1 = {2}, error = {3}".format(num_iterations, b0, b1, compute_cost_for_line_given_points(b0, b1, points)))
num_iterations = 3000

learning_rates = [0.000008, 0.000005, 0.00001]



initial_b0 = np.random.randn(1,1)*0.01

initial_b1 = np.random.randn(1,1)*0.01



for lr in learning_rates:

    _, cost_history = gradient_descent(points, initial_b0, initial_b1, lr, num_iterations)

    plt.plot(cost_history, linewidth=2)



plt.title("Gradient descent with different learning rates", fontsize=16)

plt.xlabel("number of iterations", fontsize=14)

plt.ylabel("cost", fontsize=14)

plt.legend(list(map(str, learning_rates)))
learning_rate = 0.025

num_iterations = 200



initial_b0 = np.random.randn(1,1)*0.01

initial_b1 = np.random.randn(1,1)*0.01



_, cost_history = gradient_descent(points, initial_b0, initial_b1, learning_rate, num_iterations)



plt.plot(cost_history, linewidth=2)

plt.title("Gradient descent with learning rate = " + str(learning_rate), fontsize=12)

plt.xlabel("number of iterations", fontsize=14)

plt.ylabel("cost", fontsize=14)

plt.grid()

plt.show()