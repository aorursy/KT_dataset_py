# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/weather_madrid_LEMD_1997_2015.csv')
data.info()
data.columns = [each.split()[0]+"_"+each.split()[1] if (len(each.split()) > 1) else each for each in data.columns]

data.rename(columns = {" CloudCover":"CloudCover"," Events":"Events"}, inplace = True)
data.corr()
f,ax = plt.subplots(figsize = (20,20))

sns.heatmap(data.corr(),annot = True, linewidths = 0.5, fmt = ".1f", ax = ax, cmap = "Blues")

plt.show()
data.head(10)
data.columns
data.CloudCover.plot(kind = 'line', color = "green", label = "Cloud Cover", figsize = (15,8),

                        linewidth = 1, alpha = 0.5, grid = True, linestyle = "-")



data.Precipitationmm.plot(kind = 'line', color = "green", label = "Precipitation (mm)", figsize = (15,8),

                            linewidth = 1, alpha = 1, grid = True, linestyle = "-")

plt.legend()

plt.xlabel("X-axis")

plt.ylabel("Y-axis")

plt.title("Line Plot")

plt.show()
data.plot(kind = "scatter", x = "Mean_TemperatureC", y = "Mean_Humidity", color = "red", 

          alpha = .8, figsize = (15,8))

plt.xlabel("Mean Temperature")

plt.ylabel("Mean Humidity")

plt.title("Mean Temperature - Mean Humidity Scatter Plot")

plt.show()
data.Dew_PointC.plot(kind = "hist", bins = 10, figsize = (8,8), color = "r", alpha = 0.7)

plt.title("Dew Point")

plt.show()

data.CloudCover.plot(kind = "hist", bins = 50)

plt.clf()
dictionary = {"Feynman" : "Istatistik", "Akkaya" : "RadarTemelleri"}

print(dictionary.keys())

print(dictionary.values())
dictionary['Feynman'] = "FizikDersleri"

print(dictionary)

dictionary['Nisanyan'] = "EasternTurkey"

print(dictionary)

del dictionary['Akkaya']

print(dictionary)

print('Feynman' in dictionary)

dictionary.clear()

print(dictionary)
series = data["Min_TemperatureC"]

print(type(series))

frame = data[["Min_TemperatureC"]]

print(type(frame))
x = data["Max_TemperatureC"] > 30

data[x]
data[np.logical_and(data['Max_TemperatureC'] > 30, data['Min_TemperatureC'] > 25)]
data[(data["Max_TemperatureC"] > 30) & (data["Min_TemperatureC"] > 25)]
for index,value in data[['Dew_PointC']][0:5].iterrows():

    print(index," : ",value)
n = int(input("Enter the number:"))



def cube():

    """this function calculates the cube of numbers"""    

    s = n**3

    return s

print("Square of number: " , cube())
x = 5 

def s():

    x = 25 

    return x

print("The global x: " , x)

print("The local x: " , s())
x = 5

def g():

    y = x**2 # there is no local scope so it will use the global one.

    return y

print(g())
import builtins

dir(builtins)
from math import sqrt

x = 9 # global scope

def f1():

    x = 4 # enclosed scope

    z = sqrt(x) # enclosed scope

    def f2():

        y = z * x # local scope

        return y

    return x + f2() + z

print(f1())
def g(a,b,c = 4): 

    y = a + b + c

    return y

print(g(9,13)) # it uses c variable's default value

#change default argument

print(g(9,13,6))
def h(*args):

    for i in args:

        print(i)

h(5)



print(" ")



h(5,0,5,0)



print("")



def f(**kwargs):

    for key, value in kwargs.items():

        print(key, ":", value)

        

f(harper_lee = "to kill a mockingbird", lutgens_et_al= "essentials of geology")
cube = lambda x:x**3

print(cube(4))
list1 = [4,5,7,10]

square = map(lambda x:x**2,list1)

print(list(square))