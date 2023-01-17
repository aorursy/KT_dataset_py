# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/vgsales.csv')
df.info()
df.columns

df.head(10) # The first 10 lines in our data set to look at quickly.
df.tail(10) # The last 10 lines are shown
df.dtypes  # Shows the types of columns.
df.describe()  # We'll get some of mathematical and statical results with this function.
df.corr()  # Correlation between the features.
# HeatMap

f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt='.2f', ax=ax)

# Scatter plot
# x = year y = Japan Sales

df.plot(kind='scatter', x='Year', y='JP_Sales', alpha=0.5, color='blue', grid=True)
plt.xlabel("Year")
plt.ylabel("Japan Sales")
plt.title("Year - Japan Sales Plot")
plt.show()
# Histogram plot

df.Year.plot(kind='hist', bins=100, figsize=(12, 12))
plt.show()
