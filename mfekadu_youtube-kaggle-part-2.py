# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualizations

import matplotlib.pyplot as plt # visualizations

from scipy import stats # stats.mode and stats.norm

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# df is the dataframe object from pandas
df = pd.read_csv("../input/USvideos.csv")

# display the first 10 rows of df
df.head(10)
print("here are the top 5 most viewed videos")
print(df.sort_values(['views'], ascending=False)[['views', 'title']][:5])
print("\n"+"="*70+"\n")

print("looks like over 200 million views")

print("\n"+"="*70+"\n")
print("This plot below just shows a histogram of videos with < 10 million views")
print("Of course, there are less videos with 1M views than 200K views")

plt.hist(x=df[df.views < 10000000]['views'], bins=50)
plt.title(s="Views Histogram")
plt.show()

vals = np.random.exponential(scale=1e6, size=10000)
plt.hist(x=vals,bins=50)
plt.title(s="Random Exponential Histogram")
plt.show()
percentile50 = np.percentile(a=df[df.views < 10000000]['views'], q=50)
print("The 50th percentile of views = ", percentile50)

percentile90 = np.percentile(a=df[df.views < 10000000]['views'], q=90)
print("The 90th percentile of views = ", percentile90)

percentile20 = np.percentile(a=df[df.views < 10000000]['views'], q=20)
print("The 20th percentile of views = ", percentile20)

percentile99 = np.percentile(a=df[df.views < 10000000]['views'], q=99)
print("The 99th percentile of views = ", percentile99)

percentile1 = np.percentile(a=df[df.views < 10000000]['views'], q=1)
print("The 1st percentile of views = ", percentile1)
np.mean(df[df.views < 10000000]['views'])
np.var(df[df.views < 10000000]['views'])
stats.skew(df[df.views < 10000000]['views'])
stats.kurtosis(df[df.views < 10000000]['views'])
x = df['views']
y = df['likes']
plt.scatter(x,y)
plt.title("Views vs Likes")
plt.xlabel("Views")
plt.ylabel("Likes")
plt.show()

x = np.random.randn(500)
y = np.random.randn(500)
plt.scatter(x,y)
plt.title("random scatter plot")
plt.show()