# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
%matplotlib inline
sns.set(color_codes=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read the input file
data = pd.read_csv('../input/ks-projects-201801.csv')
#Basic info
data.info()
#First 10 rows
data.head(10)
top_cat = data.main_category.value_counts()
x = top_cat.index.tolist()
y = top_cat

plt.figure(figsize=(12, 8)) #make it bigger
plt.yticks(np.arange(len(x)),x, fontsize = 16)
plt.gca().invert_yaxis() #make rank 1 show at the top
plt.barh(np.arange(len(x)),y)
plt.title('Top categories', fontsize = 20)
plt.xlabel('Number of campaigns', fontsize = 16)
win_cat = data.loc[data.state=='successful','main_category'].value_counts()
x = win_cat.index.tolist()
y = win_cat

plt.figure(figsize=(12, 8))
plt.yticks(np.arange(len(x)),x, fontsize = 16)
plt.gca().invert_yaxis()
plt.barh(np.arange(len(x)),y)
plt.title('Winning categories', fontsize = 20)
plt.xlabel('Number of campaigns', fontsize = 16)
success_prob = data.loc[data.state=='successful','main_category'].value_counts()/data.main_category.value_counts()
x = success_prob.sort_values().index.tolist()
y = success_prob.sort_values()

plt.figure(figsize=(12, 8))
plt.yticks(np.arange(len(x)),x, fontsize = 16)
plt.barh(np.arange(len(x)),y)
plt.title('Categories with highest chances of success', fontsize = 20)
plt.xlabel('Odds of success', fontsize = 16)
results = data.state.value_counts()
#Excluding live, undefined and suspended
plt.pie(results[:3], autopct='%1.1f%%',pctdistance = 1.2, startangle = 90, explode = (0,0.05,0), shadow = True)
plt.legend(labels = results.index.tolist(), loc="best")
plt.axis('equal')
plt.title('Campaign results', fontsize = 16, y=1.08)
plt.tight_layout()
#Some functions to format axis labels
def millions(x, pos):
    'The two args are the value and tick position'
    return '$%1.0fM' % (x*1e-6)
formatter = FuncFormatter(millions)

def thousands(x, pos):
    'The two args are the value and tick position'
    return '%dk' % (x*1e-3)
formatter2 = FuncFormatter(thousands)

def millionsdec(x, pos):
    'The two args are the value and tick position'
    return '$%1.2fM' % (x*1e-6)
formatter3 = FuncFormatter(millionsdec)
plt.figure(figsize=(12, 8))
plt.plot(np.arange(len(data.usd_goal_real)),data.usd_goal_real.sort_values())
plt.ticklabel_format(style='plain', axis='y')
plt.title('Goal value', fontsize = 24)
plt.xlabel('Campaigns', fontsize = 16)
plt.ylabel('Goal amount ($ USD)', fontsize = 16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax = plt.gca()
ax.get_yaxis().set_major_formatter(formatter)
ax.get_xaxis().set_major_formatter(formatter2)
plt.figure(figsize=(12, 8))
plt.plot(np.arange(len(data.usd_goal_real)),data.usd_goal_real.sort_values())
plt.ylim(0,1000000)
plt.ticklabel_format(style='plain', axis='y')
plt.title('Goal value', fontsize = 24)
plt.xlabel('Campaigns', fontsize = 16)
plt.ylabel('Goal amount ($ USD)', fontsize = 16)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
ax = plt.gca()
ax.get_yaxis().set_major_formatter(formatter3)
ax.get_xaxis().set_major_formatter(formatter2)
print("Average goal is : $%d" % data.usd_goal_real.mean())
print("Median goal is : $%d" % data.usd_goal_real.median())
data.sort_values(by = 'usd_goal_real', ascending=False)[:10]
succ_goal = data.loc[data.state=='successful','usd_goal_real'] #Goals set by successful campaigns
fail_goal = data.loc[data.state=='failed','usd_goal_real']
y1 = succ_goal.sort_values(ascending=False)
x1 = np.arange(len(y1))
y2 = fail_goal.sort_values(ascending=False)
x2 = np.arange(len(y2))
fig, ax = plt.subplots(2,figsize=(12,8))
ax[0].plot(x1,y1)
ax[0].set_title('Success', fontsize = 20)
ax[0].yaxis.set_major_formatter(formatter3)
ax[0].set_xlim(0,100)
ax[0].tick_params(axis='both',labelsize=16)

ax[1].plot(x2,y2)
ax[1].set_title('Fail', fontsize = 20)
ax[1].yaxis.set_major_formatter(formatter)
ax[1].set_xlim(0,100)
ax[1].tick_params(axis='both',labelsize=16)

plt.tight_layout()
avgamt = data.usd_pledged_real/data.backers
with pd.option_context('mode.use_inf_as_null', True):
    avgamt = avgamt.dropna()
medianamt = avgamt.median()
print("Median pledged amount is : $%1.2f" %medianamt)
averageamt = avgamt.mean()
print("Average pledged amount is : $%1.2f" %averageamt)
maxamt = avgamt.max()
print("Max pledged amount is : $%1.2f" %maxamt)