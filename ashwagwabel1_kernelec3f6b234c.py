import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir(os.getcwd()))





%matplotlib inline

plt.style.use("seaborn")
# Import data #

dpath = '../input/diamonds.csv'

diamonddf = pd.read_csv(dpath)
diamonddf.head()
diamonddf.info()
diamonddf.drop('clarity', axis=1, inplace=True)
diamonddf
diamonddf.isna().sum()
colors = sns.color_palette("deep")

sns.distplot(diamonddf["carat"], color = colors[0])

plt.show()

# 
colors = sns.color_palette("deep")

sns.distplot(diamonddf["depth"], color = colors[0])

plt.show()
colors = sns.color_palette("deep")

sns.distplot(diamonddf["table"], color = colors[0])

plt.show()
colors = sns.color_palette("deep")

sns.distplot(diamonddf["price"], color = colors[0])

plt.show()
diamonddf["price"].mean()
diamonddf["price"].mode()
 