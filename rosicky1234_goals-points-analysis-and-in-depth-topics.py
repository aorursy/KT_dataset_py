# Link as below: 
# https://www.kaggle.com/slehkyi/extended-football-stats-for-european-leagues-xg
import pprint
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
overall = pd.read_csv('../input/extended-football-stats-for-european-leagues-xg/understat.com.csv')
overall = overall.rename(index=int, columns={'Unnamed: 0': 'league', 'Unnamed: 1': 'year'}) 

match_info = pd.read_csv('../input/extended-football-stats-for-european-leagues-xg/understat_per_game.csv')
print(match_info.dtypes)
overall['avg_miss'] = overall['missed']/overall['matches']
overall['avg_goal'] = overall['scored']/overall['matches']
overall['avg_pts'] = overall['pts']/overall['matches']
# Find the number of matches played and the total goals
matches = match_info.groupby(['league', 'year']).agg({ 'h_a':'count' , 'scored':'sum'}).reset_index()
matches = DataFrame(matches.rename(index=int, columns={'h_a': 'matches'}))

# Dividing number of matches by 2 because one match will appear for two times
matches['matches'] = matches['matches']/2
matches['avg_goal'] = matches['scored']/ matches['matches'] 
print(matches)
# Visualization of Average League Goals
plt.figure(figsize=(15,10))

for l in set(matches['league']):
    temp = matches[matches['league']==l]
    x = temp['year']
    y = temp['avg_goal']
    plt.plot(x,y,label=l)
    plt.legend(loc='best')
    
# Top 20 by number of goals
groupby = overall.groupby(['league', 'year','team']).mean()['avg_goal']
groupby= groupby.nlargest(20)
color = plt.cm.RdYlGn(np.linspace(0,1,20))
groupby.plot.bar(color = color)

# Top 20 Teams by Goals Missed 
most_miss = overall.groupby(['league', 'year','team']).mean()['avg_miss']
most_miss = most_miss.nlargest(20)
print(most_miss)
color = plt.cm.RdYlGn(np.linspace(0,1,20))
groupby.plot.bar(color = color)

overall_subset = overall[['league','scored','missed','ppda_coef','oppda_coef','deep','deep_allowed']]
corrmat = overall_subset.corr()
plt.figure(figsize=(10,10))
g = sns.heatmap(corrmat,annot=True,cmap="RdYlGn")
#overall_subset = overall[['scored','missed','ppda_coef','oppda_coef','deep','deep_allowed']]
for l in set(overall_subset['league']):
    df = overall_subset[overall_subset['league']==l]
    corrmat = df.corr()
    plt.figure(figsize=(10,10))
    ax = plt.axes()
    ax.set_title(l)
    g = sns.heatmap(corrmat,annot=True,cmap="RdYlGn")

subsets = ['scored','xG','ppda_coef','deep','pts']
RFPL = match_info[match_info['league']=='RFPL']
RFPL = RFPL[subsets]
Other = match_info[match_info['league']!='RFPL']
Other = Other[subsets]

for i in ['scored','xG','ppda_coef','deep']:
    print('Average',i,'in RFPL =',round(RFPL[i].mean(),3)*2,)
    print('Average',i,'in Other Leagues =',round(Other[i].mean(),3)*2,)
    print('\n')
std_list = [None] * 6
league = [None] * 6
i = 0
for l in set(overall['league']):
    df = overall[overall['league']==l]
    print('Average std of points in ',l,'=',round(df['pts'].std(),3))
    std_list[i] = round(df['pts'].std(),3)
    league[i] = l
    i = i+1
    
plt.figure(figsize=(10,8))
plt.plot(league,std_list)
plt.legend(loc='best')
plt.xlabel('League')
plt.ylabel('Standard Deviation of League Points Got')

Paris = overall[overall['team']=='Paris Saint Germain']


fig = plt.figure(figsize = (10, 5)) 
y_axis = ""
title = ""
# creating the bar plot 
for i in ['pts','scored','missed']:
    plt.bar(Paris['year'], Paris[i]) 
    plt.xlabel("Year")
    if i == 'pts': 
        y_axis = "No. of Points Achieved";
        title = "Points achieved by Paris Saint Germain"
    elif i =='scored':
        y_axis = "No. of Scores Achieved";
        title = "Goals achieved by Paris Saint Germain"
    else:
        y_axis = "No. of Goals Missed"
        title = "Goals missed by Paris Saint Germain"
    plt.ylabel(y_axis) 
    plt.title(title) 
    plt.show() 