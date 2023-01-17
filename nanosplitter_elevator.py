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
def custom_plot(cData, cPlt, kind, figSize, x, y, xlabel, ylabel, title):

    cData.plot(kind=kind, figsize=figSize, x=x, y=y)

    cPlt.xlabel(xlabel, size = 40, color="grey")

    cPlt.ylabel(ylabel, size = 30, color="grey")

    cPlt.title(title, size = 60, color="grey")

    cPlt.xticks(size = 50, color="grey")

    cPlt.yticks(size = 60, color="grey")

    cPlt.show()

    

def basic_plot(plt, figsize, x, y, xlabel, ylabel, title):

    plt.figure(figsize=(80,15))

    plt.xlabel(xlabel, size = 40, color="grey")

    plt.ylabel(ylabel, size = 40, color="grey")

    plt.title(title, size = 60, color="grey")

    plt.xticks(size = 50, color="grey")

    plt.yticks(size = 60, color="grey")



    plt.scatter(x, y)

    plt.show()
TIMESTAMP = 0

XACC = 1

YACC = 2

ZACC = 3
f = open("/kaggle/input/elevator-acceleration/ElevatorAcc.txt", "r")
data = [list(map(float, x.split()[1:])) for x in list(filter(lambda x: "ACCEL" in x, f.readlines()))]

d = []

for i in data:

    temp = dict()

    temp["time"] = i[TIMESTAMP]

    temp["xacc"] = (i[XACC] + 1) * 9.80665

    temp["yacc"] = (i[YACC] + 1) * 9.80665

    temp["zacc"] = (i[ZACC] + 1) * 9.80665

    d.append(temp)



data = pd.DataFrame(d)



data["time"] = data["time"][50000:175000]

data["xacc"] = data["xacc"][50000:175000]

data["yacc"] = data["yacc"][50000:175000]

data["zacc"] = data["zacc"][50000:175000]

custom_plot(cData = data, cPlt = plt, kind = "scatter", figSize = (80,15), x = "time", y = "xacc", xlabel = 'Time (s)', ylabel = 'Acceleration x (m/s^2)', title = 'Acceleration X Over Time')
custom_plot(cData = data, cPlt = plt, kind = "scatter", figSize = (80,15), x = "time", y = "yacc", xlabel = 'Time (s)', ylabel = 'Acceleration y (m/s^2)', title = 'Acceleration Y Over Time')
custom_plot(cData = data, cPlt = plt, kind = "scatter", figSize = (80,15), x = "time", y = "zacc", xlabel = 'Time (s)', ylabel = 'Acceleration z (m/s^2)', title = 'Acceleration Z Over Time')
xrange = frange(2, 5, 0.001)

def pureNoise(x):

    return x + rand.random() / 7 * rand.randrange(-1, 2)

def artifData(x):

    return (math.exp(-6*(x-3)**2) - math.exp(-6*(x-4)**2)) + rand.random() / 7 * rand.randrange(-1, 2)



def smoothDerivPlot(bandwidth, xRange, plt, plotFlag):

    y = [artifData(i) for i in xRange]

    yNoise = [pureNoise(i) for i in xRange]

    yNoiseSmoothed = sci.savgol_filter(yNoise, bandwidth, 2)

    ySmoothed = sci.savgol_filter(y, bandwidth, 2)

    yDeriv = np.diff(ySmoothed) / np.diff(xRange)

    if plotFlag:

        basic_plot(plt=plt, figsize=(80,15), x=xRange, y=ySmoothed, xlabel="X", ylabel="Y", title="Smoothed Funciton - Window: " + str(bandwidth))

        basic_plot(plt=plt, figsize=(80,15), x=xRange[1:], y=yDeriv, xlabel="X", ylabel="Y", title="Derivative of Smoothed Function - Window: " + str(bandwidth))

        basic_plot(plt=plt, figsize=(80,15), x=xRange, y=yNoiseSmoothed, xlabel="X", ylabel="Y", title="Smoothed Pure noise - Window: " + str(bandwidth))

        print("-----------")

for i in range(3, 3000, 502):

    smoothDerivPlot(i, xrange, plt, True)
data["smoothZ"] = sci.savgol_filter([i for i in data["zacc"].replace(np.NaN, 0)], 501, 2)
#Plot OG data

custom_plot(cData = data, cPlt = plt, kind = "scatter", figSize = (80,10), x = "time", y = "zacc", xlabel = 'Time (s)', ylabel = 'Acceleration z (m/s^2)', title = 'Acceleration Z Unsmoothed')



#Plot window 51 smoothed data SavGol

custom_plot(cData = data, cPlt = plt, kind = "scatter", figSize = (80,10), x = "time", y = "smoothZ", xlabel = 'Time (s)', ylabel = 'Acceleration z (m/s^2)', title = 'Acceleration Z SavGol Smoothed')





#



zoomTime = [i for i in data["time"]][50000:100000]

zoomData = [i for i in data["zacc"]][50000:100000]

zoomDataSmooth = [i for i in data["smoothZ"]][50000:100000]



#

basic_plot(plt=plt, figsize=(80,10), x=zoomTime, y=zoomData, xlabel="Time (s)", ylabel="Acceleration Z (m/s^2)", title="Acceleration Z SavGol Unsmoothed Zoomed")



#

basic_plot(plt=plt, figsize=(80,10), x=zoomTime, y=zoomDataSmooth, xlabel="Time (s)", ylabel="Acceleration Z (m/s^2)", title="Acceleration Z SavGol Smoothed Zoomed")
y = [math.sin(x) for x in frange(0, 10, 0.001)]



time = [i for i in frange(0, 10, 0.001)]



derivTest = np.diff(y)/np.diff(time)



basic_plot(plt=plt, figsize=(80,15), x=time, y=y, xlabel="Time (s)", ylabel="sin(x)", title="sin(x)")



#



basic_plot(plt=plt, figsize=(80,15), x=time[1:], y=derivTest, xlabel="Time (s)", ylabel="sin(x) Prime", title="sin(x) Prime")
absDeriv = []

smoothAcc = [i for i in data["smoothZ"]]



time = [i for i in data["time"]][1:]



absDeriv = np.diff(smoothAcc[1:])/np.diff(time)
basic_plot(plt=plt, figsize=(80,30), x=time[1:], y=absDeriv, xlabel="Time (s)", ylabel="Jerk (m/(m/s^2)^2)", title="Jerk of Elevator Over Time")



zoomDataDeriv = absDeriv[50000:100000]



#

basic_plot(plt=plt, figsize=(80,30), x=zoomTime, y=zoomDataDeriv, xlabel="Time (s)", ylabel="Jerk (m/(m/s^2)^2)", title="Jerk of Elevator Over Time Zoomed")