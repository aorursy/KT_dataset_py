import numpy as np # Linear algebra

import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Graphing

import scipy.signal as sci # Data smoothing

import math # General math functions

import random as rand # For randomness
def frange(x, y, jump):

    l = []

    i = x

    while i < y:

        l.append(i)

        i += jump

    return l

data = pd.read_csv("/kaggle/input/bus-acceleration-data/bus_data.csv")
data.plot(kind="line", figsize=(80,30), x="Time (s)", y="Absolute acceleration (m/s^2)")

plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('Absolute Acceleration (m/s^2)', size = 30, color="grey")

plt.title('Absolute Acceleration Over Time', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")

plt.show()
data.plot(kind="line", figsize=(80,30), x="Time (s)", y="Acceleration z (m/s^2)")

plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('Acceleration z (m/s^2)', size = 30, color="grey")

plt.title('Absolute Acceleration Z Over Time', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")

plt.show()
data.plot(kind="line", figsize=(80,30), x="Time (s)", y="Acceleration x (m/s^2)")

plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('Acceleration x (m/s^2)', size = 30, color="grey")

plt.title('Absolute Acceleration X Over Time', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")

plt.show()
data.plot(kind="line", figsize=(80,30), x="Time (s)", y="Acceleration y (m/s^2)")

plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('Acceleration y (m/s^2)', size = 30, color="grey")

plt.title('Absolute Acceleration Y Over Time', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")

plt.show()
xrange = frange(0, 10, 0.001)

def triangleWave(x):

    return math.sin(x) + rand.random() * rand.randrange(-2, 2)



print(len(xrange))



triangle = [triangleWave(i) for i in xrange]



triangleSmooth = sci.savgol_filter(triangle, 5001, 2)



deriv = np.diff(triangleSmooth) / np.diff(xrange)





#





plt.figure(figsize=(80,15))

plt.xlabel('X', size = 40, color="grey")

plt.ylabel('Y', size = 40, color="grey")

plt.title('Original Function', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.scatter(xrange, triangle)

plt.show()



#





plt.figure(figsize=(80,15))

plt.xlabel('X', size = 40, color="grey")

plt.ylabel('Y', size = 40, color="grey")

plt.title('Smoothed Function', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.scatter(xrange, triangleSmooth)

plt.show()



#



plt.figure(figsize=(80,15))

plt.xlabel('X', size = 40, color="grey")

plt.ylabel('Y', size = 40, color="grey")

plt.title('Derivative of Smoothed Function', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.scatter(xrange[1:], deriv)

plt.show()
data["abs acc smooth sav"] = sci.savgol_filter([i for i in data["Absolute acceleration (m/s^2)"].replace(np.NaN, 0)], 51, 2)
#Plot OG data



data.plot(kind="scatter", figsize=(80,15), x="Time (s)", y="Absolute acceleration (m/s^2)", marker=",")



plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('Absolute Acceleration (m/s^2)', size = 30, color="grey")

plt.title('Absolute Acceleration Unsmoothed', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.show()



#Plot window 51 smoothed data SavGol



data.plot(kind="scatter", figsize=(80,15), x="Time (s)", y="abs acc smooth sav")



plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('Absolute Acceleration (m/s^2)', size = 30, color="grey")

plt.title('Absolute Acceleration SavGol Smoothed', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.show()
y = [math.sin(x) for x in frange(0, 10, 0.001)]



time = [i for i in frange(0, 10, 0.001)]



derivTest = np.diff(y)/np.diff(time)





plt.figure(figsize=(80,15))

plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('x^2', size = 40, color="grey")

plt.title('x^2', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.scatter(time, y)

plt.show()



#





plt.figure(figsize=(80,15))

plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('x^2 Prime', size = 40, color="grey")

plt.title('x^2 Prime', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.scatter(time[1:], derivTest)

plt.show()
absDeriv = []

smoothAcc = [i for i in data["abs acc smooth sav"]]



time = [i for i in data["Time (s)"]][1:]



absDeriv = np.diff(smoothAcc[1:])/np.diff(time)
plt.figure(figsize=(80,15))

plt.xlabel('Time (s)', size = 40, color="grey")

plt.ylabel('Jerk (m/(m/s^2)^2)', size = 40, color="grey")

plt.title('Jerk of Bus Ride Over Time', size = 60, color="grey")

plt.xticks(size = 50, color="grey")

plt.yticks(size = 60, color="grey")



plt.scatter(time[1:], absDeriv)

plt.show()