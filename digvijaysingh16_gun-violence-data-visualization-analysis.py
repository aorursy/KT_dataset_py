#Data manipulation modules
import pandas as pd        # R-like data manipulation
import numpy as np         # n-dimensional arrays - Base Package

#For pltting
import matplotlib.pyplot as plt      # For base plotting
# Seaborn is a library for making statistical graphics
# in Python. It is built on top of matplotlib and 
#  numpy and pandas data structures.
import seaborn as sns                # Easier plotting

#Misc - Will use for reading data from file
import os
#Set working directory
os.chdir("/Users/digvijay/Desktop/study/BigData/kaggle/GunViolence")
os.listdir()
gunD = pd.read_csv("../input/gun-violence-data_01-2013_03-2018.csv")
type(gunD)
gunD.shape
gunD.info()
gunD.columns
gunD.head(10)
gunD['state'].unique()
grouped = gunD.groupby('state').size().reset_index(name='counts').sort_values(by=('counts'),ascending=False)
grouped.counts = grouped.counts.astype(int)
grouped.info()
plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot1 = sns.barplot(x="state", y="counts", data=grouped)

plot1.set_xticklabels(grouped['state'], rotation=90, ha="center")
plot1.set(xlabel='States',ylabel='Counts')
plot1.set_title('Gun violance incidents recorded state wise')
plt.show()
plt.figure(figsize = (20,20))
labels = grouped.state
plt.pie(grouped.counts, autopct='%1.1f%%')
plt.legend(labels, loc="best")
plt.title('Goal Score', fontsize = 20)
plt.axis('equal')
plt.show()
gunD['date'] = pd.to_datetime(gunD['date'])
gunD['month'] = gunD.date.map(lambda date: date.month)
gunD['year'] = gunD.date.map(lambda date: date.year)
gunD.head(10)
grouped = gunD.groupby('state')
g=grouped['n_killed'].agg([np.sum]).reset_index().sort_values(by=('sum'),ascending=False)
g=pd.DataFrame(g)
g
g.info()
g.shape
plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot2 = sns.barplot(x="state", y="sum", data=g)

plot2.set_xticklabels(g['state'], rotation=90, ha="center")
plot2.set(xlabel='States',ylabel='sum')
plot2.set_title('People killed state wise')
plt.show()
grouped = gunD.groupby('state')
g=grouped['n_injured'].agg([np.sum]).reset_index().sort_values(by=('sum'),ascending=False)
plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot3 = sns.barplot(x="state", y="sum", data=g)

plot3.set_xticklabels(g['state'], rotation=90, ha="center")
plot3.set(xlabel='States',ylabel='sum')
plot3.set_title('People injured state wise')
plt.show()
grouped = gunD.groupby('state')
g=grouped['n_injured','n_killed'].agg([np.sum]).reset_index()
g.shape
g.head()
g.n_injured.shape
g.plot(x="state", y=["n_injured", "n_killed"], kind="bar")
plt.figure(figsize = (20, 20), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.0)
plot3 = g.plot(x="state", y=["n_injured", "n_killed"], kind="bar")

plot3.set_xticklabels(g['state'], rotation=90, ha="center")
plot3.set(xlabel='States',ylabel='sum')
plot3.set_title('People injured and killed state wise')
plt.show()
grouped = gunD.groupby(['state','city_or_county']).size().reset_index(name='counts').sort_values(by=('counts'),ascending=False)
g=grouped.loc[grouped.state=="Illinois",:]
g=pd.DataFrame(g)
g.columns
g = g.loc[g.counts>10,['city_or_county','counts']]
g.head(30)
plt.figure(figsize = (12, 10), facecolor = None)
sns.set_style("darkgrid")
sns.set(font_scale=1.5)
plot3 = sns.barplot(x="city_or_county", y="counts", data=g)

plot3.set_xticklabels(g['city_or_county'], rotation=90, ha="center")
plot3.set(xlabel='City or conuty',ylabel='counts')
plot3.set_title('Crimes reported in City/County of Illinois')
plt.show()
plt.figure(figsize = (10,10))
labels = g.city_or_county
plt.pie(g.counts, autopct='%1.1f%%')
plt.legend(labels, loc="best")
plt.title('Goal Score', fontsize = 20)
plt.axis('equal')
plt.show()
