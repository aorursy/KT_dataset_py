# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

# airport_passengers.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df = pd.read_csv('/kaggle/input/number-of-taiwan-airport-passengers-per-month/airport_passengers.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'airport_passengers.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head()
df["總計"].plot.hist()

plt.show()
df["臺北松山機場"].plot.hist()

plt.show()
df["臺東機場"].plot.box()

plt.show()
sns.pairplot(df, x_vars=['桃園國際機場'], y_vars='澎湖機場', markers="+", size=4)

plt.show()
dfcorr=df.corr()

dfcorr
sns.heatmap(dfcorr,annot=True,cmap='pink')

plt.show()
fig, axes = plt.subplots(1, 1, figsize=(12, 4))

sns.boxplot(x='高雄國際機場', y='臺中機場', data=df, showfliers=False);
g = sns.jointplot(x="臺北松山機場", y="七美機場", data=df, kind="kde", color="m")

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("$臺北松山機場$", "$七美機場$");
import matplotlib.style



import matplotlib as mpl



mpl.style.use('classic')
sns.jointplot(df['北竿機場'],df['金門機場'],data=df,kind='scatter')
sns.jointplot(df['南竿機場'],df['蘭嶼機場'],data=df,kind='kde',space=0,color='g')
fig=sns.jointplot(x='恆春機場',y='七美機場',kind='hex',color= 'orange', data=df)
g = (sns.jointplot("臺中機場", "臺東機場",data=df, color="r").plot_joint(sns.kdeplot, zorder=0, n_levels=6))
ax= sns.boxplot(x="高雄國際機場", y="臺中機場", data=df)

ax= sns.stripplot(x="高雄國際機場", y="臺中機場", data=df, jitter=True, edgecolor="gray")



boxtwo = ax.artists[2]

boxtwo.set_facecolor('yellow')

boxtwo.set_edgecolor('black')

boxthree=ax.artists[1]

boxthree.set_facecolor('red')

boxthree.set_edgecolor('black')

boxthree=ax.artists[0]

boxthree.set_facecolor('green')

boxthree.set_edgecolor('black')



plt.show()
sns.set(style="darkgrid")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig = sns.swarmplot(x="臺北松山機場", y="綠島機場", data=df)
fig=sns.lmplot(x="花蓮機場", y="臺南機場",data=df)
# venn2

from matplotlib_venn import venn2

花蓮機場 = df.iloc[:,0]

臺東機場 = df.iloc[:,1]

澎湖機場 = df.iloc[:,2]

臺南機場 = df.iloc[:,3]

# First way to call the 2 group Venn diagram

venn2(subsets = (len(花蓮機場)-15, len(臺東機場)-15, 15), set_labels = ('花蓮機場', '臺東機場'))

plt.show()
df.plot.area(y=['總計','桃園國際機場','臺北松山機場','嘉義機場'],alpha=0.4,figsize=(12, 6));