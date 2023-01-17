# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from scipy.stats import ttest_ind

from scipy.stats import probplot

import matplotlib.pyplot as plt

import pylab

cereals = pd.read_csv("../input/cereal.csv")

cereals.head()



probplot(cereals["sodium"],dist="norm",plot=pylab,fit="true")
hot_cereals = cereals["sodium"][cereals["type"] == 'H']

cold_cereals = cereals["sodium"][cereals["type"] == 'C']

ttest_ind(hot_cereals,cold_cereals,equal_var = False)
print("Mean  for Hold DRINKS")

print(hot_cereals.mean())

print("Mean for Cold DRINKS")

print(cold_cereals.mean())
plt.hist(hot_cereals,alpha=0.5,label = "cold",color="blue")

plt.hist(cold_cereals,label="hot",color="yellow")

plt.title("Sodium rates")

plt.legend(loc="upper left")
