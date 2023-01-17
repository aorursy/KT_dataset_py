# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind # that includes only the ttest_ind function fromscipy.stats library.(obviously I'll use it for t test) 
from scipy.stats import probplot # that includes only the probplot function fromscipy.stats library. (I'll use it for testing the distrubition)
import pylab # test distrubition 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

cereals = pd.read_csv("../input/cereal.csv")
# cereals.describe()
sugar = cereals["sugars"]
#plt.hist(sugar, edgecolor = "black") 

# to see if the sugar variable disturbuted normally 
# probplot(sugar, dist="norm", plot=pylab)

#get sugar for hot cereals
hotCereals = sugar[cereals["type"] == "H"]
#get sugar for cold cereals
coldCereals = sugar[cereals["type"] == "C"]
#compare
ttest_ind(hotCereals, coldCereals, equal_var=False)
# alpha = 0.05 we would reject the null (i.e. can be pretty sure that there's not not a difference between these two groups).


print(hotCereals.mean())
print(coldCereals.mean())

plt.hist(coldCereals, alpha= 0.8, label='cold') # alpha sets the opacity.
plt.hist(hotCereals, label='hot')

# I am just getting fancy in here
# add a legend to upper right
plt.legend(loc='upper right')
# add a title
plt.title("Sugar content of cereals by type")
