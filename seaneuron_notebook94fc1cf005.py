import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab 

import scipy.stats as stats
df = pd.read_csv("../input/data.csv",header = 0)

df.head()
df.iloc[:,1:].describe()
normalised = (df["radius_mean"] - df["radius_mean"].mean())/df["radius_mean"].std()

d = {"radius_mean": normalised, "diagnosis": df["diagnosis"]}

norm_df = pd.DataFrame(data = d)

norm_df.head()
stats.probplot(norm_df["radius_mean"], dist="norm", plot=pylab)

pylab.show()
print(stats.shapiro(df["radius_mean"]))

print(stats.shapiro(normalised))
stats.wilcoxon(df[df["diagnosis"] == 'M']["radius_mean"].sample(n=200) \

               , df[df["diagnosis"] == 'B']["radius_mean"].sample(n=200))