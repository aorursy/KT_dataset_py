# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-coronavirus-romania/covid-19RO.csv')

df.dataframeName = 'covid-19RO.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.tail()
df.info()
plt.plot(df.cases)
plt.plot(df.cases.diff())
def logistic(x, L, k, x0):

    return L / (1 + np.exp(-k * (x - x0))) + 1
p0 = [80000, 0.2, 90]



popt, pcov = curve_fit(logistic, df.index, df.cases, p0, method = "trf")

print("Last day number of cases: " + str(int(df.cases[-1:])))

print("Number of cases aproximated for the next day: " + str(int(int(df.cases[-1:] + logistic(len(df) , *popt) - logistic(len(df)-1 , *popt)))))



plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days")

plt.plot(df.index, df.cases, 'b-', label='data')

plt.plot(range(len(df) + 1), logistic(range(len(df) + 1), *popt), 'r-', label='fit')
print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))

print("Predicted k (growth rate): " + str(float(popt[1])))

print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")
plt.title("Number of cases (blue) and prediction (red) for " + str(int(popt[-1]*2)) + " days")

plt.plot(df.index, df.cases, 'b-', label='data')

plt.plot(range(int(popt[-1]*2)), logistic(range(int(popt[-1]*2)), *popt), 'r-', label='fit')
plt.title("Log scale of cases (blue) and prediction (red) for " + str(len(df.cases)+10) + " days")

plt.plot(np.log(df.cases[10:]), 'b-')

plt.plot(np.log(logistic(range(len(df.cases)+10), *popt)), 'r-')
def logisticFixedx0(x, L, k):

    x0 = 67

    return L / (1 + np.exp(-k * (x - x0))) + 1
p0 = [80000, 0.2]



popt, pcov = curve_fit(logisticFixedx0, df.index, df.cases, p0, method = "trf")

print("Last day number of cases: " + str(int(df.cases[-1:])))

print("Number of cases aproximated for the next day: " + str(int(int(df.cases[-1:] + logisticFixedx0(len(df) , *popt) - logisticFixedx0(len(df)-1 , *popt)))))



plt.title("Number of cases (blue) and prediction (red) for " + str(len(df)) + " days, with inflexion in day 67")

plt.plot(df.index, df.cases, 'b-', label='data')

plt.plot(range(len(df) + 1), [logisticFixedx0(x , *popt) for x in range(len(df)+1)], 'r-', label='fit')
print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))

print("Predicted k (growth rate): " + str(float(popt[1])))
plt.plot(df.cases.diff())
def gaussian(x, a, b, c):

    return a * np.exp(-(1/2.)* np.power(((x - b)*1.) / c, 2))
plt.plot([gaussian(i, 380, 55, 20) for i in range(180)], 'r-', label='fit')

plt.plot(df.index, df.cases.diff(), 'b-', label='data')
plt.title("Number of tests per day")

plt.plot(df.tests)
plt.title("Number of new tests and confirmed cases per day")

plt.plot(df.tests.diff())

plt.plot(df.cases.diff()*10)
from sklearn.linear_model import LinearRegression





a = np.array(df.cases.diff()[2:])

b = np.array(df.tests.diff()[:-2])



X = np.array(range(len(a))).reshape((len(a), 1))

y = np.nan_to_num(a/b)



reg = LinearRegression().fit(X, y)



yhat = reg.predict(X)



plt.title("Percent of confirmed cases out of performed tests per day, with 2 days delayed result")

plt.scatter(range(len(a)), a/b)

plt.plot(range(len(a)), yhat, color='blue', linewidth=3)
X = df.index

a = np.array(df.cases.diff())

y = df.cases * (1 + np.array(reg.predict(np.array(range(len(a))).reshape((len(a), 1)))))



p0 = [80000, 0.2, 80]



popt, pcov = curve_fit(logistic, X, y, p0, method = "lm")

print("Last day number of cases: " + str(int(y[-1:]/(1 + reg.predict([[len(X)-1]])[0]))))

print("Number of cases aproximated for the next day: " + str(int(int(

                                                                     int(

                                                                         y[-1:]/(1 + reg.predict([[len(X)-1]])[0]))

                                                                     + 

                                                                     int(

                                                                         logistic(len(X) , *popt)/(1 + reg.predict([[len(X)]])[0]))

                                                                     - int(

                                                                         logistic(len(X)-1 , *popt)/(1 + reg.predict([[len(X)]])[0]))

                                                                 ))))



plt.title("Number of cases (blue) and prediction (red) for " + str(len(X)) + " days")

plt.plot(X, y, 'b-', label='data')

plt.plot(range(len(X) + 1), logistic(range(len(X) + 1), *popt), 'r-', label='fit')
print("Predicted L (the maximum number of confirmed cases): " + str(int(popt[0])))

print("Predicted k (growth rate): " + str(float(popt[1])))

print("Predicted x0 (the day of the inflexion): " + str(int(popt[2])) + "")