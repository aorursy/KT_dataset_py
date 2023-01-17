# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

from tables import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np



import matplotlib.pyplot as plt



data = np.array([ [0.3, 5.8], [2.2, 4.4], [0.5, 6.5], [0.7, 5.8],

    [1.0, 5.6], [1.8, 5.0], [3.0, 4.8], [0.2, 6.0], [2.3, 6.1]])



# Sources:

# - https://en.wikipedia.org/wiki/Simple_linear_regression



def calc_rxy():

    pass



class Estimator:

    def __init__(self, data):

        self.data = data

        

        X = data[:,0]

        Y = data[:,1]

        print("X:", X, "Y:", Y)

        

        X_squared = [x**2 for x in X]

        Y_squared = [y**2 for y in Y]

        print("X_squared:", np.sum(X_squared))

        print("Y_squared:", np.sum(Y_squared))

        

        avg_x = np.sum(X) / len(data)

        avg_y = np.sum(Y) / len(data)

        avg_xy = np.sum(X * Y) / len(data)

        print("avg_x:", avg_x, "avg_y:", avg_y, "avg_xy:", avg_xy)

        

        rxy = (avg_xy - avg_x * avg_y) / np.sqrt((np.sum(X_squared) / len(data) - avg_x**2) * (np.sum(Y_squared) / len(data) - avg_y**2))

        

        print("rxy:", rxy)

        print("linear erklärt:", rxy **2 * 100, "%")

        

        sx = np.sqrt(np.sum([(x - avg_x)**2 for x in X]) / len(data))

        sy = np.sqrt(np.sum([(y - avg_y)**2 for y in Y]) / len(data))

        

        print("sx:", sx, "sy:", sy)

        

        self.beta = rxy * sy / sx

        print("beta:", self.beta)

        

        self.alpha = avg_y - self.beta * avg_x

        print("alpha:", self.alpha)

        

    def estimate(self, key):

        return self.alpha + self.beta * key

    

estimator = Estimator(data)



X_estimate = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

Y_estimated = [estimator.estimate(x) for x in X_estimate]



X = data[:,0]

Y = data[:,1]

Y_error = []

Y_pred = [estimator.estimate(x) for x in X]



print("Y:", Y)

print("predicted:", Y_pred)



for i in range (0, len(X)):

    Y_error.append(estimator.estimate(X[i]) - Y[i])

print("errors:", Y_error)



avg_error = np.sum(Y_error) / len(data)

print("avg_error for training set:", avg_error)



variance_error = np.sum([(err - avg_error)**2 for err in Y_error]) / (len(Y_error) - 1)

print("variance_error for training set:", variance_error)



print("max_error:", max(Y_error))



plt.scatter(X_estimate,Y_estimated)

plt.scatter(X,Y)

#plt.scatter(X,Y_pred)

plt.show()



from tabulate import tabulate

print(tabulate(zip(X_estimate, Y_estimated), headers=["X", "Y"]))


# Y and Y_pred distribution plots.

sns.distplot(Y);

sns.distplot(Y_pred)