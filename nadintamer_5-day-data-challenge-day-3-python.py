#import necessary libraries

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

from scipy.stats import ttest_ind



#read dataset and type/sodium content of cereal

cereal_data = pd.read_csv("../input/cereal.csv")

hotcold = cereal_data["type"]

sodium = cereal_data["sodium"]



#separate sodium values into hot or cold

hot = []

cold = []

for x in range(len(hotcold)):

    if hotcold[x] == "C":

        cold.append(sodium[x])

    elif hotcold[x] == "H":

        hot.append(sodium[x])



#calculate standard deviation for both groups

#since it is not equal, use equal_var = False

print(np.std(hot))

print(np.std(cold))



#perform t-test

ttest_ind(hot, cold, equal_var = False)
#plot histogram of hot values

plt.hist(hot, edgecolor="black")

plt.title("Sodium Content in Hot Cereals") # add a title

plt.xlabel("Sodium in milligrams") # label the x axes 

plt.ylabel("Count") # label the y axes
#plot histogram of cold values

plt.hist(cold, edgecolor="black")

plt.title("Sodium Content in Cold Cereals") # add a title

plt.xlabel("Sodium in milligrams") # label the x axes 

plt.ylabel("Count") # label the y axes