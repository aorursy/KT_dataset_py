import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data = pd.read_csv("../input/data_set_ALL_AML_train.csv")

data.describe()

#data.describe(include = "all") That's for ALL information
print(data.columns) #List all the column names
for i in range(31):

    plt.hist(data[str(i+1)],edgecolor="black")

    plt.xlabel("position")

    plt.ylabel("weight")

plt.title("Data Achieved :)")
#There is a slight difference between PLT and PANDAPLT:

#Whether to plot one graph or multiple graphs

for i in range(31):

    data.hist(column = str(i+1),edgecolor="black")

    plt.title("Data " + str(i+1) + " Achieved :)")

    plt.xlabel("position")

    plt.ylabel("weight")
print("So this is Day 2 :)")