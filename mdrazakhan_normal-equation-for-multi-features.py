import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv('../input/linear-regression-dataset/cars.csv')

df = df.dropna()

x = df.iloc[:, 2:4].values

y = df.iloc[:, 4].values
x_bias = np.ones((int(x.size/x[0].size), 1))

x_updated = np.append(x_bias, x, axis = 1)
def normalEq(x):

    x_transpose = np.transpose(x)

    x_transpose_dot_x = x_transpose.dot(x)

    temp_1 = np.linalg.pinv(x_transpose_dot_x)

    temp_2 = x_transpose.dot(y)

    theta = temp_1.dot(temp_2)

    

    print(theta)

    

    x1_predict = float(input('Enter Volume '))

    x2_predict = float(input('Enter Weight '))

    



    y_predict = theta[0] + theta[1]*x1_predict + theta[2]*x2_predict

    

    print(y_predict)



    
normalEq(x_updated)