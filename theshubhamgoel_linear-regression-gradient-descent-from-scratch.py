#Load data from file

import pandas as pd



def loadData(path, sep=',', headerIndex=0):

    df = pd.read_csv(path, sep=sep, header=headerIndex)

    matrix = df.values

    number_of_columns = df.columns.size

    X = matrix[:, 0:number_of_columns - 1]

    y = matrix[:, [number_of_columns - 1]]



    return X, y
#compute cost

def computeCost(X, y, theta):

    m = np.shape(y)[0]



    J = (1 / (2 * m)) * np.sum(np.power((np.matmul(X, theta) - y), 2))

    return J
#compute Gradient

def computeGradiant(X, y, theta):

    m = np.shape(y)[0]



    gradiant = (1 / m) * (np.matmul(X.T, np.matmul(X, theta) - y))

    #print("Gradiant value : ", gradiant)

    return gradiant
#This is simple gradiant desent algorithm. Here Goal is to minimize the cost function over no of iteration.

def gradiant_desent_algo(X,

                         y,

                         learningRate=.01,

                         iteration=100,

                         should_print_cost=True):



    #Initalize vector theata as zero

    theta = np.zeros((X.shape[1], 1))

    

    J = computeCost(X, y, theta)

    print("Initial cost : ", J)



    #J_History can be used to visualize how cost changs over every iteration

    J_History = np.full((iteration + 1, 1), J)



    for i in range(iteration):

        gradiant = computeGradiant(X, y, theta)

        theta = theta - learningRate * gradiant

        

        #Optionally caluate J_History for visualization

        J = computeCost(X, y, theta)

        J_History.itemset((i + 1, 0), J)

        if should_print_cost:

            print("Cost after iteration ", (i + 1), " : ", J)



    print("Final cost : ", J)

    return theta, J_History
def gradiant_decent_solution(X, y, iterations = 10000):

    #adding extra column for 0th index

    new_X = np.append(np.ones((X.shape[0], 1)), X, axis=1)



    theta, J_History = gradiant_desent_algo(

        new_X, y, learningRate=.01, iteration=iterations, should_print_cost=False)

    

    # Equation for 1 feature and 1 bias

    # y = mx + c

    m = theta[1]

    c = theta[0]

    print("Value of m : ", m)

    print("Value of c : ", c)



    #show J_history

    plotCostFunction(J_History)



    #plot data

    plotData(X, y, "Linear Regression with Gradient Descent.")

    abline(m, c)
#noraml equation technique

import numpy as np

from numpy.linalg import inv, pinv



def calculate_theta_with_normal_eqaution(X, y):

    theta = np.matmul(np.matmul(inv(np.matmul(X.T, X)), X.T), y)

    #or

    theta = np.matmul(pinv(X), y)

    # hypothesis = x0*theta0 + x1*theta1 + x2*theta2 + ...

    return theta
def noraml_equation_solution(X, y):

    new_X = np.append(np.ones((X.shape[0], 1)), X, axis=1)

    theta = calculate_theta_with_normal_eqaution(new_X, y)



    # Equation for 1 feature and 1 bias

    # y = mx + c

    m = theta[1]

    c = theta[0]

    print("Value of m : ", m)

    print("Value of c : ", c)



    #plot data

    plotData(X, y, "Linear Regression with Normal Equation.")

    abline(m, c)
#Visualizting cost function, data and prediction 

%matplotlib inline

import matplotlib.pyplot as plt



def plotData(X, y, title, markersize=12):

    plt.plot(X, y, 'b+', markersize=markersize)

    #plt.axis([0, 10, 0, 10])

    X_label = plt.xlabel('X Values')

    Y_label = plt.ylabel('Y Values')

    X_label.set_color("#FFFFFF")

    Y_label.set_color("#FFFFFF")

    plt.title(title)



def abline(slope, intercept):

    """Plot a line from slope and intercept"""

    axes = plt.gca()

    x_vals = np.array(axes.get_xlim())

    y_vals = intercept + slope * x_vals

    plt.plot(x_vals, y_vals, 'g--')

    plt.show()



def plotCostFunction(J_history):

    X = np.arange(J_history.shape[0])

    y = J_history

    plt.plot(X, y, 'b--')

    X_label = plt.xlabel('Iteration')

    Y_label = plt.ylabel('Cost')

    X_label.set_color("#FFFFFF")

    Y_label.set_color("#FFFFFF")

    plt.title('Cost function changes over no of iterations')

    plt.show()
#main

#X, y = loadData('../input/perfect_points.csv')

X, y = loadData('../input/points_with_small_noise.csv')

noraml_equation_solution(X,y)
gradiant_decent_solution(X, y, iterations=100)
gradiant_decent_solution(X, y, iterations=100000)