# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/salary-data-simple-linear-regression/Salary_Data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values
def mean(d): return sum(d)/len(d)
def std_dev(d):

    d_mean = mean(d)

    numerator = 0

    for e in d:

        numerator += (e-d_mean)**2

    return (numerator/(len(d)-1))**(1/2)
def corr_coeff(X, y):

    n = len(X)

    Xy = []

    for i in range(n):

        Xy.append(X[i]*y[i])

    X_sq = [e**2 for e in X]

    y_sq = [e**2 for e in y]

    numerator = n*sum(Xy)-sum(X)*sum(y)

    denominator = ((n*sum(X_sq)-sum(X)**2)*(n*sum(y_sq)-sum(y)**2))**(1/2)

    return numerator/denominator
def slope(r, std_dev_X, std_dev_y): return r*(std_dev_y/std_dev_X)
def y_intercept(slope, X_mean, y_mean): return y_mean-(slope*X_mean)
def equation_coeffs(X, y):

    b1 = slope(corr_coeff(X, y), std_dev(X), std_dev(y))[0]

    b0 = y_intercept(b1, mean(X), mean(y))[0]

    return b0, b1
def estimate(e_X):

    return b0 + b1*e_X
def equation():

    return 'y=' + str(b0) + ('+' if b1 >= 0 else '') + str(b1) + 'x'
def r_squared():

    ss_res = sum([(y[i]-estimate(X[i]))**2 for i in range(len(X))])

    ss_tot = sum([(y[i]-mean(y))**2 for i in range(len(X))])

    return 1-(ss_res/ss_tot)
def plot(eq, r_sq):

    title = str(eq) + ' : R^2=' + str(r_sq)

    plt.title(title)

    plt.xlabel('INDEPENDANT')

    plt.ylabel('DEPENDANT')

    plt.scatter(X, y, color='red')

    plt.plot(X, [estimate(e) for e in X], color='blue')

    plt.show()
b0, b1 = equation_coeffs(X, y)
plot(equation(), r_squared())