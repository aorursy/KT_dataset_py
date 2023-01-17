# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

df = pd.read_csv('../input/data.csv', header = None)

df.head()
from numpy import *



def compute_error_for_line_given_points(b, m, points):

    totalError = 0

    for i in range(0, len(points)):

        x = points[i, 0]

        y = points[i, 1]

        totalError += (y - (m*x + b)) ** 2

        return totalError / float(len(points))



def step_gradient(b_current, m_current, points, learningRate):

    # gradient descent

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

    return [b,m]



def run():

    points = genfromtxt('../input/data.csv', delimiter = ',')

    #hyperparameters

    learning_rate = 0.0001

    # y = mx + b

    initial_b = 0 # y-intercept

    initial_m = 0 # slope of the equation

    num_iterations = 1000

    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))

    print ("Running...")

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

    

if __name__ == '__main__':

    run()
import numpy as np

X = np.array(df[0])

y = np.array(df[1])



X # Cycling distance
y # Burned Calories
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize = (15, 10))

plt.xlabel('Cycling distance')

plt.ylabel('Burned Calories')

sns.regplot(x = X, y = y)
from sklearn.linear_model import LinearRegression

X = X.reshape(-1, 1)

y = y.reshape(-1, 1)

linreg = LinearRegression()

linreg.fit(X, y)

print ("Y-intercept", linreg.intercept_[0])

print ("Slope", linreg.coef_[0][0])
from sklearn.linear_model import LinearRegression

from sklearn import metrics

import numpy as np

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1)

linreg = LinearRegression()

linreg.fit(X_train, y_train)

y_pred = linreg.predict(X_test)



print (np.sqrt(metrics.mean_squared_error(y_test, y_pred))) # RMSE