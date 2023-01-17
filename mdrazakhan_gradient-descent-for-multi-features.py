import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv('../input/linear-regression-dataset/cars.csv')

df.head()

print(df.mean())
X1 = (df['Volume']-df['Volume'].mean())/df['Volume'].max()

X2 = (df['Weight']-df['Volume'].mean())/df['Weight'].max()



y = (df['CO2']-df['CO2'].mean())/df['CO2'].max()

def j(theta_0, theta_1, theta_2):

    bloop = 0

    for k in range(35):

        bloop += (theta_0 + theta_1*X1[k] + theta_2*X2[k] - y[k])**(2)

    return bloop/2*36



def djd0(theta_0, theta_1, theta_2):

    bloop = 0

    for j in range(35):

        bloop += theta_0 + theta_1*X1[j] + theta_2*X2[j] - y[j]

    return bloop/36



def djd1(theta_0, theta_1, theta_2):

    bloop = 0

    for j in range(35):

        bloop += (theta_0 + theta_1*X1[j] + theta_2*X2[j] - y[j])*X1[j]

    return bloop/36



def djd2(theta_0, theta_1, theta_2):

    bloop = 0

    for j in range(35):

        bloop += (theta_0 + theta_1*X1[j] + theta_2*X2[j] - y[j])*X2[j]

    return bloop/36

         
def gradient_descent(theta0_start, theta1_start, theta2_start, learning_rate):

     

    # These x and y value lists will be used later for visualization.

    

    theta0_grad = [theta0_start]

    theta1_grad = [theta1_start]

    theta2_grad = [theta2_start]

    j_grad = [j(theta0_start, theta1_start, theta2_start)]

    

    while True:

         

        # Get the Slope value from the derivative function for x_start

        # Since we need negative descent (towards minimum), we use '-' of derivative

        theta0_start_derivative = djd0(theta0_start, theta1_start, theta2_start)

        theta1_start_derivative = djd1(theta0_start, theta1_start, theta2_start)

        theta2_start_derivative = djd2(theta0_start, theta1_start, theta2_start)



        

        # calculate x_start by adding the previous value to 

        # the product of the derivative and the learning rate calculated above.

        theta0_start -= (learning_rate * theta0_start_derivative)

        theta1_start -= (learning_rate * theta1_start_derivative)       

        theta2_start -= (learning_rate * theta2_start_derivative)       



        

        theta0_grad.append(theta0_start)

        theta1_grad.append(theta1_start)

        theta2_grad.append(theta2_start)



        j_grad.append(j(theta0_start, theta1_start, theta2_start))

        if(j_grad[-1] >= j_grad[-2]):

            learning_rate /= 3

            continue

        

        if(j_grad[-2]-j_grad[-1] <= 10**(-6)):

            break

        

    print("Learning Rate: ",learning_rate)



    print("Minimun occurs at",theta0_start, theta1_start, theta2_start)

        

        

        

    plt.scatter(X1,y)

    plt.scatter(X2,y)

    plt.show()

    

    print ("Number of steps: ",len(theta0_grad)-1)

    x1_predict = float(input('Volume'))

    x2_predict = float(input('Weight'))

    x1_predict = (x1_predict-df['Volume'].mean())/df['Volume'].max()

    x2_predict = (x2_predict-df['Weight'].mean())/df['Weight'].max()



    y_predict = theta0_start+theta1_start*x1_predict+theta2_start*x2_predict

    

    y_predict = y_predict*df['CO2'].max()+df['CO2'].mean()

    

    print('CO2', y_predict)
gradient_descent(0, 0, 0, 2)