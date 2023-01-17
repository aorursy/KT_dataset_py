# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

cereal_data = pd.read_csv("../input/cereal.csv")

temp = cereal_data["type"]

sugar = cereal_data["sugars"]



hot = []

cold = []

for x in range(len(temp)):

    if temp[x] == "C":

        cold.append(sugar[x])

    elif temp[x] == "H":

        hot.append(sugar[x])

        

print(np.std(hot))

print(np.std(cold))



print(ttest_ind(hot, cold, equal_var = False))



plt.hist(hot, edgecolor="black")

plt.title("Sugar Content in Hot Cereals")

plt.xlabel("Sugar") 

plt.ylabel("Number")







#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
plt.hist(cold, edgecolor="black")

plt.title("Sugar Content in Cold Cereals")

plt.xlabel("Sugar") 

plt.ylabel("Number")