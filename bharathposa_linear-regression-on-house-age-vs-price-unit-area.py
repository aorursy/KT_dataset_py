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
#Read the dataset



data_csv = pd.read_csv('../input/Real estate.csv', error_bad_lines=False)

data_csv.head(10)
# Plotting the dataset

import matplotlib.pyplot as plt

x = data_csv['X2 house age']

y = data_csv['Y house price of unit area']





#Converting them into numpy array

X = np.array(x)

Y = np.array(y)



plt.scatter(X,Y)

plt.show()
#initialize m and c

m = 0

c = 0
# taking iterations of 10000

iterations = 10000
#setting the learning rate

learning_rate = 0.001
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

    

print(loss_values[-1])

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