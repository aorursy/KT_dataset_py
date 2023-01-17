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
data = pd.read_csv("../input/insurance.csv")
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot = True, linewidth = .5, fmt = ".1f", ax = ax)

plt.show()
data.head(10)
# Line Plot

data.bmi.plot(kind = 'line', color = 'g',label = 'bmi',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.charges.plot(color = 'r',label = 'charges',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc = "upper right")

plt.xlabel("x bmi")

plt.ylabel("y charges")

plt.title("Line Plot")

plt.show()
# Scatter Plot 

data.plot(kind = "scatter",x="bmi",y = "charges",alpha = 0.5,color="red")

plt.xlabel("bmi")

plt.ylabel("charges")

plt.title("Bmi Charges Scatter Plot")

#Histogram

data.bmi.plot(kind = "hist",bins = 50,figsize =(12,12))

plt.show()
# clf

data.bmi.plot(kind="hist",bins = 50)

plt.clf()
# dictionary 

dictionary = {"germany" : "Hamburg", "Netherland" : "Amsterdam", "Italy" : "Roma"}

print(dictionary.keys())

print(dictionary.values())
dictionary["germany"] = "Frankfurt"

print(dictionary)

dictionary["greece"] = "Atina"

print(dictionary)

del dictionary["germany"]

print(dictionary)

print("bulgaria" in dictionary)

dictionary.clear()

print(dictionary)
print(dictionary)
series = data["bmi"]

print(type(series))

data_frame = data[["bmi"]]

print(type(data_frame))
# Comparison operator

print(3 > 2)

print(3 != 2)

# Boolean operators

print(True and False)

print(True or False)
x = data["bmi"]>29.000

data[x]
data[np.logical_and(data["bmi"]>28.000, data["charges"]>1137.00000)]
data[(data["bmi"]>28.000) & (data["charges"]>1137.00000)]
i = 0

while i != 5:

    print("i is:" ,i)

    i += 1

print("i is : equal to 5")
lis = [1,2,3,4,5]

for i in lis:

    print("i is : ",i)

print("")



for index, value in enumerate(lis):

    print(index, " : ", value)

print("")




