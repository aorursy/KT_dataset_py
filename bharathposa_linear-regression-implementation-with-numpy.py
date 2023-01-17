#Importing numpy library

import numpy as np
#Download Dataset

import urllib

urllib.request.urlretrieve("https://raw.githubusercontent.com/llSourcell/Intro_to_the_Math_of_intelligence/master/data.csv","data.csv")
#Read the dataset

import pandas as pd

headers = ['X','Y']

data_csv = pd.read_csv('data.csv', error_bad_lines=False, names=headers)

data_csv.head(10)
# Plotting the dataset

import matplotlib.pyplot as plt

x = data_csv['X']

y = data_csv['Y']





#Converting them into numpy array

X = np.array(x)

Y = np.array(y)



plt.scatter(X,Y)

plt.show()

#initialize m and c

m = 0

c = 0



# taking iterations of 10000

iterations = 100
#setting the learning rate

learning_rate = 0.0001
def caliculate_error(x, y, m, c):

    y_predicted = m * x + c 

    error = (1/2)*((y - y_predicted)**2)

    error = np.mean(error)

    return  error



    
def update_value(x,m,c,y, learning_rate):

    m_gradient = -x*(y - (m*x + c))

    c_gradient = -(y - (m*x+c))

    m_new  = m - (learning_rate*m_gradient)

    c_new  = c - (learning_rate*c_gradient)



    return np.mean(m_new), np.mean(c_new)
loss_values = []
# Extract M and C values 

for i in range(iterations):

    loss_values.append(caliculate_error(X,Y,m,c))

    m_new, c_new = update_value(X,m,c,Y, learning_rate)

    m = m_new

    c = c_new

    

print(m)

print(c)
#plotting the loss_values



plt.plot(loss_values)

plt.show()
#plot line with the updated values 



Y_latest = (X*m + c)



plt.scatter(X,Y_latest)

plt.scatter(X,Y)



plt.show()