import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



df = pd.read_csv("/kaggle/input/train.csv")

df = df.dropna()

df = df.reset_index(drop=True)



X = df['x']/10

y = df['y']/10  

    

    
plt.scatter(X, y)



x = np.linspace(X.min(), X.max(), 10000)



plt.show()
def j(x):

    bloop = 0

    for i in range(X.size):

        bloop += (x*X[i] - y[i])**(2)

        

    return bloop/(2*10)



def j_derivative(x):

    bloop = 0

    for i in range(X.size):

        bloop += (x*X[i] - y[i])*X[i]

        

    return bloop/10
def plot_gradient(x, y, x_vis, y_vis):

    plt.subplot(1,2,2)

    plt.scatter(x_vis, y_vis, c = "b")

    plt.plot(x, j(x), c = "r")

    plt.title("Gradient Descent")

    plt.show()

     

def gradient_descent(x_start, learning_rate):

     

    # These x and y value lists will be used later for visualization.

    x_grad = [x_start]

    y_grad = [j(x_start)]

    

    while True:

         

        # Get the Slope value from the derivative function for x_start

        x_start_derivative = j_derivative(x_start)

        # calculate x_start by adding the previous value to 

        # the product of the derivative and the learning rate calculated above.

        x_start -= (learning_rate * x_start_derivative)        

        

        x_grad.append(x_start)

        y_grad.append(j(x_start))

        

        if(y_grad[-1] >= y_grad[-2]):

            learning_rate /= 3

            continue

        

        if(y_grad[-2]-y_grad[-1] <= 10**(-5)):

            break

        

         

    plt.plot(x, x_start*x,c = 'r')

    plt.scatter(X,y)

    plt.show()

    plot_gradient(x, j(x) ,x_grad, y_grad)

    print ("Local minimum occurs at: {:.2f}".format(x_start))

    print ("Number of steps: ",len(x_grad)-1)

    m=x_start

    x_predict = float(input())

    print(m*x_predict)

    
gradient_descent(0, 1)

