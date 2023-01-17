# This is a code cell
# Execute this cell with shift + enter

print("Es hat funktioniert")
# Kommentar

print("Hallo")

print(5)
# All Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

from mpl_toolkits.mplot3d import Axes3D # 3D visualization

import os # file system access
def print_some_things(things, num):

    '''

    ---------------------------------------------

    This function prints things a number of times

    

    things --> array/list of objects to print



    num --> integer: amount of times to print

    

    ---------------------------------------------

    '''

    

    # Loop over num

    for i in range(num):

        # print the content of things to console

        print(i, "\t",things)
things = [1, 2, 3, 42]

print_some_things(things, 42)
import pandas as pd



#define Data and structure

columns = ["OEM", "Modell", "Price", "TFlops"]



data = [

    ["Apple", "Macbook Pro", 2200, 1],

    ["Apple", "Macbook", 1400, 0.8],

    ["Dell", "XPS 15", 1900, 1.1],

    ["Lenovo", "Yoga", 1100, 0.6],

    ["HP", "Elitebook", 1600, 0.9],

    ["HP", "X360", 1900, 1.1],

    ["Asus", "Zephyrus", 2900, 1.5],

    ["Microsoft", "Surface Pro", 1800, 0.9]

]



# build dataframe

dataframe = pd.DataFrame(columns = columns, data = data)



# display dataframe

dataframe.head(10)
# only show OEMs

dataframe.OEM
# count how often every OEM is included

dataframe.OEM.value_counts()
import matplotlib.pyplot as plt # data visualization

# define data to plot

x = dataframe.Price

y = dataframe.TFlops



# create canvas

fig, ax = plt.subplots()



# plot data

ax.scatter(x, y)



# plot figure to console

plt.show()
# Random Values for Plot

x = np.random.uniform(1,0,200)

y = np.random.uniform(1,0,200)

z = np.random.uniform(1,0,200)



x2 = np.random.uniform(1.5,0,200)

y2 = np.random.uniform(1.5,0,200)

z2 = x2 + y2



# create canvas

fig, ax = plt.subplots()



# plot data

ax.scatter(x, y)



# set some visualizations

ax.set_xlabel("x", fontsize=15)

ax.set_ylabel("y", fontsize=15)

ax.set_title('random plot')



# plot figure to console

plt.show()
# create 3D canvas

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')



# plot data

ax.scatter(x, y, z, c="blue")

ax.scatter(x2, y2, z2, c = "red")



# name axis

ax.set_xlabel("x", fontsize=15)

ax.set_ylabel("y", fontsize=15)

ax.set_zlabel("z", fontsize=15)

ax.set_title('also random')



# plot figure to console

plt.show()
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

sns.set(style="ticks")



rs = np.random.RandomState(11)

x = rs.gamma(2, size=1000)

y = -.5 * x + rs.normal(size=1000)



fig = sns.jointplot(x,y, kind="hex", color="#4CB391")

fig.set_axis_labels('Customer Retention time', 'Willingness to rcommend', fontsize=16)

fig
# 42 zu ersetzen:

a1 = [1, 42]

a2 = [2, 42]

a3 = [3, 42]

a4 = [4, 42]

a5 = [5, 42]

a6 = [6, 42]

antworten = [a1,a2,a3,a4,a5,a6]

meine_antworten = pd.DataFrame(antworten, columns = ["Id", "Category"])

meine_antworten.to_csv("meine_loesung.csv", index = False)