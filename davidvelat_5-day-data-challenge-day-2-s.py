import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt 



#day 3

from scipy.stats import ttest_ind 

from scipy.stats import probplot

import pylab 
drinks = pd.read_csv("../input/starbucks_drinkMenu_expanded.csv")

drinks
drinks.describe()
print(drinks.columns)

calories = drinks["Calories"].tolist()

print(len(calories))

print(type(calories))
# plt.hist([1,2,2,3,4,4,4,4,4])

plt.hist(calories, color="black", edgecolor = "yellow" )

drinks["Calories"].hist(color="black")