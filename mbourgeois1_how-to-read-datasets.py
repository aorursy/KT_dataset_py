

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# First, look at everything.

from subprocess import check_output

print(check_output(["ls", "../input/data"]).decode("utf8"))
#Pick a dataset

import zipfile



Dataset = "womens-world-cup-predictions"



#unzip files

with zipfile.ZipFile("../input/data/"+Dataset+".zip","r") as z:

    z.extractall(".")
from subprocess import check_output

print(check_output(["ls", "womens-world-cup-predictions"]).decode("utf8"))
#Select specific data file

d=pd.read_csv(Dataset+"/wwc-forecast-20150602-093000.csv")

d.head()
#Chapter 3--Visualizing Data

from matplotlib import pyplot as plt





#Bar Charts

team = ["USA", "GER", "JPN", "FRA", "CAN"]

winprob = [.28235, .27080, .09735, .07605, .06840]



xs = [i + 0.1 for i, _ in enumerate(team)]



plt.bar(xs, winprob)



plt.ylabel("Win Probability Percentage")

plt.title("Team Win Probability")



plt.xticks([i + 0.1 for i, _ in enumerate(team)], team)



plt.show()
#Line Charts



quarter = [.80235, .81475, .78615, .66995, .68535]

semi = [.6540, .6098, .4078, .3252, .3633]

final = [.41755, .3954, .2314, .1600, .16995]

team = ["USA", "GER", "JPN", "FRA", "CAN"]



xs = [i + 0.1 for i, _ in enumerate(team)]



plt.plot(xs, quarter, 'g-', label='quarter final prob')

plt.plot(xs, semi, 'r-.', label='semi final prob')

plt.plot(xs, final, 'b:', label='final prob')



plt.xticks([i + 0.1 for i, _ in enumerate(team)], team)



plt.legend(loc=9)

plt.xlabel("Team")

plt.title("Probability Win Differences")

plt.show()
#Chapter 5 -- Statistics 



#Mean

def mean(winprob):

    return sum(winprob)/len(winprob)



mean(winprob)
#Median

def median(v):

    n = len(v)

    sorted_v = sorted(v)

    midpoint = n // 2



    if n % 2 == 1:

        return sorted_v[midpoint]

    else: 

        lo = midpoint - 1

        hi = midpoint

        return (sorted_v[lo] + sorted_v[hi]) / 2

    

    

median(winprob)
#range

def data_range(x):

    return max(x) - min(x)



data_range(winprob)



#Chapter 4--Linear Algebre

def dot(v,w):

    """v_1 * w_1 + ... + v_n * w_n"""

    return sum(v_i * w_i

              for v_i, w_i in zip(v,w))



def sum_of_squares(v):

    """v_1 * v_1 + ... + v_n * v_n"""

    return dot(v,v)







def de_mean(x):

    x_bar = mean(x)

    return [x_i - x_bar for x_i in x]



def variance(x):

    n = len(x)

    deviations = de_mean(x)

    return sum_of_squares(deviations) / (n - 1)



variance(winprob)
#Standard Deviation 



import math



def standard_dev(x):

    return math.sqrt(variance(x))



standard_dev(winprob)
import math



def standard_deviation(x):

    return math.sqrt(variance(x))



standard_deviation(winprob)
def covariance(x,y):

    n = len(x)

    return dot(de_mean(x), de_mean(y)) / (n-1)



covariance(quarter, winprob)
#Correlation



def correlation(x,y):

    stdev_x = standard_dev(x)

    stdev_y = standard_dev(y)

    if stdev_x > 0 and stdev_y > 0:

        return covariance(x,y) / stdev_x / stdev_y

    else:

        return 0

    

    correlation(quarter, winprob)
#Correlation



def correlation(x,y):

    stdev_x = standard_deviation(x)

    stdev_y = standard_deviation(y)

    if stdev_x > 0 and stdev_y > 0:

        return covariance(x,y) / stdev_x / stdev_y

    else:

        return 0

    

    correlation(quarter, winprob)