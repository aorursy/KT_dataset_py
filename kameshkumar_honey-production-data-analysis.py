# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/honeyproduction.csv')
df.head()
df.describe()
plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.boxplot(x='year',y="yieldpercol",data=df)
plt.title('Boxplot of Production Year vs Yield Per Colony',fontsize=20)
plt.xlabel('Year',fontsize=15)
plt.ylabel('Yield Per Colony',fontsize=15)

# Inference - significant drop in mean yield from 1998 to 2012, high range of fluctuation in the period between 1998 and 2001
# but range is contained 2011 and 2012
plt.figure(figsize=(12,6))
sns.set_style('whitegrid')
sns.boxplot(x='year',y="totalprod",data=df)
plt.title('Boxplot of Production Year vs Total Production',fontsize=20)
plt.xlabel('Year',fontsize=15)
plt.ylabel('Total Production',fontsize=15)

# Inference - although mean production quantity is more or less constant, number of outliers is significant.
plt.figure(figsize=(12,6))
#sns.lmplot(x='year',y='yieldpercol',data=df)
sns.jointplot(x='year',y='yieldpercol',data=df,kind='reg')

# Linear regression plot confirms the steady decline in yield per colony over the years.
plt.figure(figsize=(12,6))
sns.barplot(x='state',y='totalprod',data=df)
plt.title('Statewise Production Trend',fontsize=20)

# ND tops the list in honey production, followed by CA and SD
# creating a summary table grouped by year to analyze the trend

by_year = df[['totalprod','year','yieldpercol','stocks','prodvalue']].groupby('year').sum()
by_year.head()
by_year.reset_index(level=0,inplace=True)
by_year.head()
#creating subplots for each of the trend analysis
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,10))

ax1.plot(by_year['year'],by_year['yieldpercol'])
ax1.set_title('Trend of Yield Per Colony',fontsize=20)
ax1.set_xlabel('Year',fontsize=15)
ax1.set_ylabel('Yield Per Colony',fontsize=15)

ax2.plot(by_year['year'],by_year['totalprod'])
ax2.set_title('Trend of Honey Production',fontsize=20)
ax2.set_xlabel('Year',fontsize=15)
ax2.set_ylabel('Production',fontsize=15)

ax3.plot(by_year['year'],by_year['stocks'])
ax3.set_title('Trend of Honey Stocks',fontsize=20)
ax3.set_xlabel('Year',fontsize=15)
ax3.set_ylabel('Stocks',fontsize=15)

ax4.plot(by_year['year'],by_year['prodvalue'])
ax4.set_title('Trend of Prod Value',fontsize=20)
ax4.set_xlabel('Year',fontsize=15)
ax4.set_ylabel('Value',fontsize=15)

plt.tight_layout()
plt.show()

# Yield per colony is showing a consistent decline trend from 2002 to 2007, there is an increase in 2008 followed by a decline in subsequent years.
# Honey production trend is also following a similar pattern due to its correlation with Yield.
# Production value is shown an upward trend, evidently the price is going up, data needs to be adjusted for inflation to check the actual trend.
# creating a state summary to find out how each state performs on the production.
by_state = df[['state','totalprod','yieldpercol']].groupby('state').sum()
by_state.reset_index(level=0,inplace=True)
by_state.head()
by_state.sort_values(by='totalprod',ascending=False,inplace=True)
by_state.head()

# ND, CA and SD are the top three states in honey production
by_state.tail()

# SC, OK, MD are the states with lowest production
plt.figure(figsize=(12,6))
sns.barplot(x='state',y='totalprod',data=by_state)
plt.title('Statewise Production Trend - Descending Order',fontsize=20)
plt.xlabel("State",fontsize=15)
plt.ylabel("Total Production",fontsize=15)
# finding out maximum production values by state
state_max = df[['state','totalprod']].groupby('state').max()
state_max.reset_index(level=0,inplace=True)
state_max.columns = ['State','Max Prod']
state_max.head()
# finding out minimum production values by state
state_min = df[['state','totalprod']].groupby('state').min()
state_min.reset_index(level=0,inplace=True)
state_min.columns = ['State','Min Prod']
state_min.head()
# merging maximum and minimum dataframe to find the range
state_range = pd.merge(state_max,state_min,how='inner',on='State')
state_range.head()
state_range['Change Percentage'] = ((state_range['Max Prod']-state_range['Min Prod'])/state_range['Max Prod'])*100
state_range.sort_values(by='Change Percentage',ascending=False,inplace=True)
state_range.head()

# MO, NM and ME are the top three state with highest decline in honey production
plt.figure(figsize=(12,6))
sns.barplot(x='State',y='Change Percentage',data= state_range)
plt.title('Statewise Production Decline Trend',fontsize=20)
plt.xlabel("State",fontsize=15)
plt.ylabel("% Decline",fontsize=15)