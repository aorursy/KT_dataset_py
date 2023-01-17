import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv("../input/cereal.csv")

df.head(3)
df.describe()
from scipy.stats import ttest_ind, probplot

import pylab

probplot(df["calories"], dist="norm", plot=pylab)


plt.hist(df["calories"], bins=10, edgecolor="black")

import matplotlib.pyplot as plt

hot = df["calories"][df["type"] == "H"]

cold = df["calories"][df["type"] == "C"]

ttest_ind(hot, cold, equal_var=False)
plt.hist(cold, bins=10, edgecolor="black")

plt.xlabel("calories")

plt.ylabel("count")

plt.title("Hot")
plt.hist(hot, bins=10, edgecolor="black")

plt.xlabel("calories")

plt.ylabel("count")

plt.title("Cold")
print(plt.hist(hot), plt.hist(cold, edgecolor="black"))
plt.hist(hot, facecolor="orange")

plt.hist(cold, facecolor="blue", edgecolor="black") #last param is resolved

plt.xlabel("calories")

plt.ylabel("count")