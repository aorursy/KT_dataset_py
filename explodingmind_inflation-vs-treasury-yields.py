# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns #a high level library for plotting

from numpy import genfromtxt

import matplotlib.pyplot as plt # low level library for plotting

from scipy import stats # for statistics like r value, slope, etc.



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print("hello world")



data = genfromtxt('../input/Inflation vs 5-year treasury bond yields - Sheet1 (1).csv', delimiter=',')



# Any results you write to the current directory are saved as output.
years = data[:, 0]

inflation = data[:, 1] * 100

yields = data[:, 2]



testHue = []



for i in range(7):

    testHue.append("1960s")



for i in range(10):

    testHue.append("1970s")



for i in range(10):

    testHue.append("1980s")



for i in range(10):

    testHue.append("1990s")



for i in range(10):

    testHue.append("2000s")



for i in range(9):

    testHue.append("2010s")

    

sns.set(style="darkgrid")

    

#we want to load our data
sns.lineplot(years, inflation)

plt.xlabel("Time (years)")

plt.ylabel("Average Inflation")

plt.title("Inflation over Time")

plt.figure()

sns.lineplot(years, yields)

plt.xlabel("Time (years)")

plt.ylabel("5-year treasury yields")

plt.title("5-year treasury yields over time")
plt.figure()

sns.scatterplot(inflation, yields, hue=testHue)

slope, intercept, r_value, p_value, std_err = stats.linregress(inflation, yields)

plt.plot(yields, intercept + slope*yields, 'r', label='fitted line', linestyle=":")

plt.title("5-year Treasury Yields vs Inflation Scatter Plot")

plt.ylabel("5-year treasury yield (Percentage Points)")

plt.xlabel("Average Annual Inflation (Percentage Points)")
plt.figure()

sns.residplot(inflation, yields)

plt.title("Residual Plot")

plt.ylabel("Residual for Predicted Value")

plt.xlabel("Inflation (Percentage Points)")
residuals = []



for i in range(np.prod(inflation.shape)):

    x = inflation[i]

    pred = x * slope + intercept

    obs = yields[i]

    resid = obs - pred

    residuals.append(resid)



plt.figure()

npp = stats.probplot(residuals, plot=plt)

plt.title("Normal Probability Plot for Residuals")

plt.ylabel("Observed Residual")

plt.xlabel("Expected Normal Residual")
plt.figure()

sns.jointplot(inflation, yields, kind="reg")

plt.xlabel("5-year treasury yield (Percentage Points)")

plt.ylabel("Average Annual Inflation (Percentage Points)")
print(f"intercept={intercept}")

print(f"p_value={p_value} note that this is the two tailed value, the actual value is half as much")

print(f"r={r_value}")

print(f"slope={slope}")

print(f"standard error={std_err}")