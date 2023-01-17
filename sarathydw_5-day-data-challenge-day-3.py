import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as scb

from scipy.stats import ttest_ind

from scipy.stats import probplot

import pylab 
df = pd.read_csv("../input/cereal.csv")

df.describe()
df.head()
hotcereals = df["sodium"][df["type"] == "H"]

coldcereals = df["sodium"][df["type"] == "C"]



ttest_ind(hotcereals, coldcereals, equal_var=False)
print("Mean sodium for hot cereals : ") 

print(hotcereals.mean())



print("Mean sodium for cold cereals : ")

print(coldcereals.mean())

#Plot the hot cereals

plt.hist(hotcereals,label='hot')

#Plot the cold cereals

plt.hist(coldcereals,alpha=0.5,label='cold')#Plot the hot cereals

#Add a legend

plt.legend(loc = 'Upper Right')

#Add a tittle

plt.title("Sodium content in Cereals by type")
probplot(df["sodium"], dist="norm", plot=pylab)
plt.hist(hotcereals,alpha=0.5,label='hot')

plt.hist(coldcereals,alpha=0.5,label='cold')

plt.legend(loc = 'Upper Right')