# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sb
from pandas import Series, DataFrame
from pylab import *
import statsmodels.api as sm
from matplotlib.ticker import MaxNLocator
from scipy import stats
import math
%matplotlib inline
plt.style.use('fivethirtyeight')

# Load Data
df = pd.read_csv('../input/Citywide_Payroll_Data__Fiscal_Year_.csv')

# Examine
df.head(n=15)
# df.tail(n=15) To see bottom of data

# Deletion 
del df["Mid Init"]
del df["First Name"]
del df["Last Name"]

# Check column variables
df.columns


print(df.shape) # 2 million rows of data. 13 Column Variables.

df.info()

# This shows us that our numerical data is in format of object not integer, we need to cast them into integers to run analysis.

# We can confirm this by running .describe() function which prints summary statistics of only numerical data
# We can see that it only outputted data for year and hours which are int/float variables. 
# this just states that our numbers aren't being read as numbers but as objects, plain variables.
# df.describe()

## Data Cleaning -
df["Title Description"].value_counts() # Some titles have non-alpha characters, we would need to clean this up as well.
# this shows us how many different types of occupations there are and how many times they occur.
df["Base Salary"] = df["Base Salary"].astype(str)
df["Regular Gross Paid"] = df["Regular Gross Paid"].astype(str)
df["Total OT Paid"] = df["Total OT Paid"].astype(str)
df["Total Other Pay"] = df["Total Other Pay"].astype(str)
# Convert "Base Salary" to Integer and rest to floats.
df["Base Salary"] = df["Base Salary"].str.replace('$','')
df["Regular Gross Paid"] = df["Regular Gross Paid"].str.replace('$','')
df["Total OT Paid"] = df["Total OT Paid"].str.replace('$','')
df["Total Other Pay"] = df["Total Other Pay"].str.replace('$','')
df["Title Description"] = df["Title Description"].astype(str)
df["Title Description"] = df["Title Description"].str.strip()
df["Regular Gross Paid"] = df["Regular Gross Paid"].str.strip()
df["Pay Basis"] = df["Pay Basis"].str.strip()
# Gets rid of Non-Alpha Characters in this column
df["Title Description"]= df["Title Description"].str.replace('[^A-Za-z\s]+', '')

#Convert them back
df["Base Salary"] = df["Base Salary"].astype(float)
df["Base Salary"] = df["Base Salary"].astype(int)
df["Regular Gross Paid"] = df["Regular Gross Paid"].astype(float)
df["Total OT Paid"] = df["Total OT Paid"].astype(float)
df["Total Other Pay"] = df["Total Other Pay"].astype(float)
df.head()
#format(a, '.2f')
#math.floor(3.1415)
df.describe()
b = df[(df["Pay Basis"] == "per Annum")]
# Subset of Data, due to pay having negative values and zeros. 
x = b[(b["Regular Gross Paid"] > 10000) & (b["Base Salary"] > 10000)]

# - Who Made the least ?  plot
x.groupby(["Fiscal Year","Title Description"])["Regular Gross Paid"].mean().sort_values(ascending=True)[:15].plot(kind='barh',color='lightslategray')
plt.xticks(fontsize=11, weight= 'bold', color='black')
plt.yticks(fontsize=9.0, weight='bold',color='black')
plt.ylabel('')
plt.xlabel('$',fontsize=9)
plt.text(x=-3000, y=16, s="Lowest Paid Employees", fontsize=28, weight='bold',alpha=.93)
plt.text(x=-3000, y=15.1, s="Employees On Annual Pay above $10,000",fontsize=8,alpha=.45)
# Who Made the most ?  plot
x.groupby(["Fiscal Year","Title Description"])["Regular Gross Paid"].mean().sort_values(ascending=False)[:15].plot(kind='barh',color='cadetblue')
plt.xticks(fontsize=11, weight= 'bold',rotation=0, color='black')
plt.yticks(fontsize=9.0, weight='bold',color='black')
plt.ylabel('')
plt.xlabel('$',fontsize=9)
plt.text(x=-180000, y=16, s="Highest Paid Employees", fontsize=28, weight='bold',alpha=.93)
plt.text(x=-180000, y=15.1, s="Employees On Annual Pay above $10,000",weight='bold',fontsize=10,alpha=.45)
plt.axvline(x=(stats.trim_mean(x['Regular Gross Paid'].values, 0.1)), color='black', linewidth=1.3, alpha=.75)
plt.annotate('Trimmed Mean',fontsize=9,xy=(90000,7.3), xytext=(68000,7.4), arrowprops=dict(arrowstyle='<-',facecolor='black',connectionstyle="arc3"))
## 2014 AGENCY EMPLOYEES
x1 = df[df["Fiscal Year"] == 2014]["Agency Name"].value_counts().sort_values(ascending=False)[0:10].values
y1 = df[df["Fiscal Year"] == 2014]["Agency Name"].value_counts().sort_values(ascending=False)[0:10].index
plt.rcParams['figure.figsize']= 9,7
ax1 = sb.barplot(x=x1, y=y1, palette='BuGn_d', saturation=1)

ax1.figure.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
sb.set_context("talk", font_scale=1.0)
sb.despine()
fontz = {'fontsize':22, 'fontweight': 'bold'}
# ax1.set(xlabel='', ylabel='')

## TOP AGENCIES WITH MOST EMPLOYEES in 2014
ax1.set_title('Top 8 Agencies With Most Employees in 2014',loc='left',fontdict=fontz, alpha=.90)
plt.text(x = 50000,y = 10.88, s="# of Employees")
plt.text(x=-45000, y=10.87, s='Justin Nunez    Source: Kaggle', fontsize=11, color='#f0f0f0',backgroundcolor='grey')
########## BEGINNING OF NESTED PLOTS

# Could use a for loop or create a function to make this data... as well as for plotting
### Data 2014 plot
x1 = df[df["Fiscal Year"] == 2014]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].values
y1 = df[df["Fiscal Year"] == 2014]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].index
### Data 2015 plot
x2 = df[df["Fiscal Year"] == 2015]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].values
y2 = df[df["Fiscal Year"] == 2015]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].index
### Data 2016 Plot
x3 = df[df["Fiscal Year"] == 2016]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].values
y3 = df[df["Fiscal Year"] == 2016]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].index
### Data 2017 plot
x4 = df[df["Fiscal Year"] == 2017]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].values
y4 = df[df["Fiscal Year"] == 2017]["Agency Name"].value_counts().sort_values(ascending=False)[0:8].index
   # Plot 
fig = plt.figure(figsize=(12,9))
ax0 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
 # titles 
fig.suptitle("Top 8 Agencies with most employees in...",fontweight='bold', fontsize=22)
ax0.title.set_text('2014')
ax2.title.set_text('2015')
ax3.title.set_text('2016')
ax4.title.set_text('2017')

sb.barplot(x=x1, y=y1, palette='BuGn_d', saturation=1, ax=ax0)
sb.barplot(x=x2, y=y2, palette='Blues_d', saturation=1, ax=ax2)
sb.barplot(x=x3, y=y3, palette='GnBu_d', saturation=1, ax=ax3)
sb.barplot(x=x4, y=y4, palette='RdBu_d', saturation=1, ax=ax4)
#plt.tight_layout(w_pad=1.1)
plt.subplots_adjust(left=.05, right=.95,hspace=.5, wspace=1.4,top=.85)
plt.text(x=80000, y=10.87, s='Justin Nunez    Source: Kaggle', fontsize=11, color='#f0f0f0',backgroundcolor='grey')
plt.show()

################################################################## END OF PLOT ABOVE
# New DataFrame With only values greater then 20,000 on per Annum Basis.
x = df[(df["Pay Basis"] == "per Annum") & (df["Regular Gross Paid"] > 20000) & (df["Base Salary"] > 20000)]

# Regular Gross Pay Plot
x.groupby(x["Fiscal Year"])['Regular Gross Paid'].median().plot(linewidth=4,c='g')
x.groupby(x["Fiscal Year"])['Regular Gross Paid'].mean().plot(linewidth=4, c='b')
plt.axhline(y=50000, color='black', linewidth=1.3, alpha=.75)
plt.xlim(left=2014, right=2016)
plt.text(x=2013.9, y=74000, s="Regular Gross Pay", fontsize=28, weight='bold',alpha=.93)
plt.text(x=2013.9, y=72600, s="Employees On Annual Pay above $20,000",fontsize=12,alpha=.75)
plt.text(x=2013.9, y=42000, s='Justin Nunez    Source: Kaggle', fontsize=11, color='#f0f0f0',backgroundcolor='grey')
plt.text(x=2016.2, y=65400, s="Median", color='g', weight='bold', rotation=7)
plt.text(x=2015.5, y=67500, s="Mean", color='b', weight='bold', rotation=7)
plt.xticks(range(2014,2018,1), fontsize=14, weight= 'bold')

# Base Salary Plot

x.groupby(x["Fiscal Year"])['Base Salary'].median().plot(linewidth=4,c='g')
x.groupby(x["Fiscal Year"])['Base Salary'].mean().plot(linewidth=4,c='b')
plt.axhline(y=50000, color='black', linewidth=1.3, alpha=.75)
plt.xlim(left=2014, right=2016)
plt.text(x=2013.9, y=75000, s="Base Salary", fontsize=26, weight='bold',alpha=.90)
plt.text(x=2013.9, y=73800, s="Employees On Annual Pay above $20,000",fontsize=12,alpha=.75)
plt.text(x=2013.9, y=42000, s='Justin Nunez    Source: Kaggle', fontsize=11, color='#f0f0f0',backgroundcolor='grey')
plt.text(x=2014, y=70600, s="Median", color='g', weight='bold', rotation=7)
plt.text(x=2014, y=67600, s="Mean", color='b', weight='bold', rotation=7)
plt.xticks(range(2014,2018,1), fontsize=14, weight= 'bold')
# Top 4 Boroughs with Highest Average Gross Pay 2014 Plot !
b = x[x["Fiscal Year"]==2014]
b.groupby(["Work Location Borough"])["Regular Gross Paid"].mean().sort_values(ascending=False)[:8].plot(kind='bar',color='darkcyan')
plt.xticks(fontsize=14, weight= 'bold',rotation=0, color='black')
plt.text(x=-.8, y=84000, s="Top 4 Boroughs with Highest Average Gross Pay in 2014", fontsize=26, weight='bold',alpha=.90)
plt.text(x=-.8, y=79600, s="Employees On Annual Pay above $20,000",fontsize=12,alpha=.75)
plt.xlabel('')
plt.axhline(y=b["Regular Gross Paid"].mean(), color='black', linewidth=1.3, alpha=.75)
plt.text(x = 2.5, y = (b["Regular Gross Paid"].mean() + 240), s="Average Gross Pay")

# Top 8 Boroughs with Highest Average Gross Pay 2015 Plot !
l = x[x["Fiscal Year"]==2015]
l.groupby(["Work Location Borough"])["Regular Gross Paid"].mean().sort_values(ascending=False)[:8].plot(kind='bar',color='darkcyan')
plt.xticks(fontsize=7.4, weight= 'bold',rotation=0, color='black')
plt.text(x=-.8, y=98000, s="Top 8 Boroughs with Highest Average Gross Pay in 2015", fontsize=26, weight='bold',alpha=.90)
plt.text(x=-.8, y=92900, s="Employees On Annual Pay above $20,000",fontsize=12,alpha=.75)
plt.xlabel('')
plt.axhline(y=l["Regular Gross Paid"].mean(), color='black', linewidth=1.3, alpha=.75)
plt.text(x = 2.5, y = (l["Regular Gross Paid"].mean() + 240), s="Average Gross Pay", fontsize=8.6)
# # Top 8 Boroughs with Highest Average Gross Pay 2016 Plot !
lg= x[x["Fiscal Year"]==2016]
lg.groupby(["Work Location Borough"])["Regular Gross Paid"].mean().sort_values(ascending=False)[:8].plot(kind='bar',color='darkcyan')
plt.xticks(fontsize=7.4, weight= 'bold',rotation=0, color='black')
plt.text(x=-.8, y=109000, s="Top 8 Boroughs with Highest Average Gross Pay in 2016", fontsize=26, weight='bold',alpha=.90)
plt.text(x=-.8, y=105000, s="Employees On Annual Pay above $20,000",fontsize=10,alpha=.75)
plt.xlabel('')
plt.axhline(y=lg["Regular Gross Paid"].mean(), color='black', linewidth=1.3, alpha=.75)
plt.text(x = 2.5, y = (lg["Regular Gross Paid"].mean() + 240), s="Average Gross Pay", fontsize=8.6)
# # Top 8 Boroughs with Highest Average Gross Pay 2017 Plot !
mg= x[x["Fiscal Year"]==2017]
mg.groupby(["Work Location Borough"])["Regular Gross Paid"].mean().sort_values(ascending=False)[:8].plot(kind='bar',color='darkcyan')
plt.xticks(fontsize=7.4, weight= 'bold',rotation=0, color='black')
plt.text(x=-.8, y=117000, s="Top 8 Boroughs with Highest Average Gross Pay in 2017", fontsize=26, weight='bold',alpha=.90)
plt.text(x=-.8, y=112000, s="Employees On Annual Pay above $20,000",fontsize=10,alpha=.75)
plt.xlabel('')
plt.axhline(y=mg["Regular Gross Paid"].mean(), color='black', linewidth=1.3, alpha=.75)
plt.text(x = 2.3, y = (mg["Regular Gross Paid"].mean() + 240), s="Average Gross Pay", fontsize=9.5)
### --- Distributions ---- ###
## creation of subets. For each year.
from scipy.stats import norm
new = df[(df["Pay Basis"] == "per Annum") & (df["Regular Gross Paid"] > 10000) & (df["Base Salary"] > 10000)]
cc = new[new["Fiscal Year"]==2014]
dd = new[new["Fiscal Year"]==2015]
ee = new[new["Fiscal Year"]==2016]
ff = new[new["Fiscal Year"]==2017]

# Year 2017 Distribution    (too lazy to add title)
sb.set_context("paper")
sb.distplot(ff["Regular Gross Paid"],fit=norm,kde=False)
#sb.set(xticks=np.arange(0,250000,50000))
plt.xlim(left=0, right=200000)
plt.title('Distribution of Gross Pay 2017')
plt.text(x=100000, y=.000013, s="Sub-set of employees with salary above 10000", fontsize=8,alpha=.65)
# Year 2017 Distribution    
sb.set_context("paper")
sb.distplot(ff["Base Salary"],fit=norm,kde=False)
#sb.set(xticks=np.arange(0,250000,50000))
plt.xlim(left=0, right=200000)
plt.title('Distribution of Base Salary 2017')
plt.text(x=100000, y=.000017, s="Sub-set of employees with salary above 10000", fontsize=8,alpha=.65)
# Distribution of Base Salary for Annual Employees making over 10,000
fig1 = plt.figure(figsize=(12,9))
ax6 = fig1.add_subplot(221)
ax7 = fig1.add_subplot(222)
ax8  = fig1.add_subplot(223)
ax9 = fig1.add_subplot(224)
 # titles 
fig1.suptitle("Base Salary Distribution (Salary > $10,000) in...",fontweight='bold', fontsize=22)
ax6.title.set_text('2014')
ax7.title.set_text('2015')
ax8.title.set_text('2016')
ax9.title.set_text('2017')
sb.distplot(cc["Base Salary"],kde=False, ax=ax6)
sb.distplot(dd["Base Salary"],kde=False, ax=ax7)
sb.distplot(ee["Base Salary"],kde=False, ax=ax8)
sb.distplot(ff["Base Salary"],kde=False, ax=ax9)
plt.subplots_adjust(left=.05, right=.95,hspace=.5, wspace=.35,top=.85)
plt.xlim(left=0, right=200000)
plt.text(x=-5000, y=-15000, s='Justin Nunez    Source: Kaggle', fontsize=9, color='#f0f0f0',backgroundcolor='grey')
# WONT RUN if I take out the "kde=False" paramter in the 4th distplot, so thats why its not on the 2017 graph.
# Distribution of Base Salary for Annual Employees making over 10,000
fig1 = plt.figure(figsize=(12,9))
ax6 = fig1.add_subplot(221)
ax7 = fig1.add_subplot(222)
ax8  = fig1.add_subplot(223)
ax9 = fig1.add_subplot(224)
 # titles 
fig1.suptitle("Base Salary Distribution (Salary > $10,000) in...",fontweight='bold', fontsize=22)
ax6.title.set_text('2014')
ax7.title.set_text('2015')
ax8.title.set_text('2016')
ax9.title.set_text('2017')
sb.distplot(cc["Base Salary"],fit=norm,kde=False, ax=ax6)
ax6.set_xlim(0,185000)
ax7.set_xlim(0,185000)
ax8.set_xlim(0,185000)
ax9.set_xlim(0,185000)
sb.distplot(dd["Base Salary"],fit=norm,kde=False, ax=ax7)
plt.xlim(left=0, right=200000)
sb.distplot(ee["Base Salary"],fit=norm,kde=False, ax=ax8)
plt.xlim(left=0, right=200000)
sb.distplot(ff["Base Salary"], kde=False, ax=ax9)
plt.subplots_adjust(left=.05, right=.95,hspace=.5, wspace=.35,top=.85)
#plt.xlim(left=0, right=175000)
plt.text(x=-5000, y=-15000, s='Justin Nunez    Source: Kaggle', fontsize=9, color='#f0f0f0',backgroundcolor='grey')
# WONT RUN if I take out the "kde=False" paramter in the 4th distplot, so thats why its not on the 2017 graph.
# distribution of Gross Pay in 2014.
sb.kdeplot(new["Regular Gross Paid"], shade=True, color="r",clip=(0,240000))
# new.describe() # Shows that 541,544.90 is Max Salary.
# histogram of 2017 Gross Pay 
ax5 = plt.subplot(111)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
#ax5.get_xaxis().tick_bottom()
#ax5.get_yaxis().tick_left()
plt.xticks(fontsize=14)
plt.text(x=1000, y= 21000, s="Distribution of Gross Pay in 2017", fontsize=27, fontweight='bold')
plt.hist(list(ff["Regular Gross Paid"].values), color="#3F5D7D", bins=100)
