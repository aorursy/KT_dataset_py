# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#To keep things simple only the data of 2015 will be looked at

d5 = pd.read_csv('../input/2015.csv')

d5.head()

d5.tail()
d5.info() 

d5.columns
d5.corr()
d5.plot(kind = "scatter", x = "Economy (GDP per Capita)", y = "Happiness Score", color = "blue", figsize = (14,14), grid = True)

plt.show()
d5["Freedom"].plot(color = "red", grid = True)

plt.show()
d5["Health (Life Expectancy)"].plot(color = "red", grid = True,)

plt.ylabel("Life Expectancy")

plt.show()
hsav = sum(d5["Happiness Score"])/len(d5["Happiness Score"])

print(hsav)

d5["Happiness Level"] = ["High" if hs > hsav else "Low" for hs in d5["Happiness Score"]]

d5.head(75)
gl = d5["Generosity"]

gav = sum(gl)/len(gl)

d5["Generosity Level"] = ["Above Average" if g > gav else "Below Average" for g in gl]

d5.head(75)
#Here we define a function to see if whether a country has both high happiness and generosity levels

#first we define two functions using the lambda function to be able to reach items in our data 

get_hl = lambda ct: d5[d5["Country"] == ct]["Happiness Level"][d5[d5["Country"] == ct].index.values[0]]

get_gl = lambda ct: d5[d5["Country"] == ct]["Generosity Level"][d5[d5["Country"] == ct].index.values[0]]

print(get_hl("Turkey"))

print(get_hl("Turkey"))
#If they are both high the function we'll define will return "Both are high"

#If only one of the two is high it'll return which one is high

#Otherwise it'll return "Both are low"

def glhl(ct):

    if(get_hl(ct) == "High"):

        if(get_gl(ct) == "Above Average"):

            return("Both are high")

        else:

            return("Happiness Level is high")

    if(get_gl == "Above Average"):

        return ("Genorosity level is high")

    return("Both are low")

print(glhl("Chile"))

print(glhl("Venezuela"))

print(glhl("Australia"))