import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #imported for visualization

from scipy.stats import ttest_ind 

from scipy.stats import probplot

import pylab



from subprocess import check_output #something defult 

print(check_output(["ls", "../input"]).decode("utf8"))#something defult 

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/cereal.csv")

df

probplot(df["calories"], dist="norm", plot=pylab)
hot = df["calories"][df["type"] == "H"] 

cold = df["calories"][df["type"]== "C"]
ttest_ind(hot,cold, equal_var=False)
print(plt.hist(cold))

print(plt.hist(hot))
print(plt.hist(hot), plt.hist(cold))