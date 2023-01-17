# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from math import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Get data csv to dataframe using pandas library
df = pd.read_csv("../input/athlete_events.csv")
print(df.columns)
df = df[pd.notnull(df['Height'])]
df = df[pd.notnull(df['Weight'])]
# Trying to give value to the cities so the name is not significant.
cities = df.get('City')
set_cities = set(cities)
values_dict_cities = dict(zip(set_cities, range(len(set_cities))))
# cities to numbers
values_cities = [values_dict_cities.get(i) for i in cities]

# Trying to give value to the sport so the name is not significant.
sports = df.get('Sport')
set_sports = set(sports)
values_dict_sports = dict(zip(set_sports, range(len(set_sports))))
# cities to numbers
values_sports = [values_dict_sports.get(i) for i in sports]
#Graphing the values
plt.scatter(values_sports,values_cities)
#Find out, cities and sports are no colinear at least by the randomenss of the dictonary sort
# and the sort of the range
# As the previous data did not work out we determine to use heigth and weight
x,y = np.array(df.get("Height"))[0:100], np.array(df.get("Weight"))[0:100]
#Graph the data to determine if the are linear
plt.scatter(x,y)
# Square sum of the errors, cost function
def cost_function(theta0, theta1):
    sum_square = ((theta0 + theta0*x)-y)**2
    return sum_square/(len(x))
#Graph cost function
print(np.mean(x+y))
space = np.linspace(100,200, num=len(x))
theta_0, theta_1 = np.meshgrid(space,space)
result = cost_function(theta_0, theta_1)

fig = plt.figure()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(theta_1, theta_0, result, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ran = int(random.random()*100)
xs, ys, zs = [theta_1[ran][ran]],[theta_0[ran][ran]],[result[ran][ran]]
ax.scatter3D(xs, ys, zs, c='r', marker='X',linewidth=2)

plt.show()

corner_masks = [False, True]
plt.contourf(theta_1, theta_0, result)
plt.plot(ys, xs, ".", c="red")

