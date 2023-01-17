# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
df_month = pd.read_csv('../input/Traffic accidents by month of occurrence 2001-2014.csv')
df_time = pd.read_csv('../input/Traffic accidents by time of occurrence 2001-2014.csv')
df_month.info()
df_time.info()
df_month.head()
total_num_of_states = df_month['STATE/UT'].unique().size
total_num_of_years = df_month['YEAR'].unique().size
print('total_num_of_states = ' + str(total_num_of_states))
print('total_num_of_years = ' + str(total_num_of_years))

avg_acdnts_per_yr_in_sts = df_month[['STATE/UT','TOTAL']].groupby('STATE/UT').sum()/total_num_of_years
avg_acdnts_per_yr_in_sts.sort_values(by='TOTAL',ascending=False,inplace=True)
fig = plt.figure()
avg_acdnts_per_yr_in_sts['TOTAL'].plot("bar", figsize = (12,6), title = 'Avg. # of Accidents per Year in Each State')
plt.ylabel('Avg # of Accidents per Year')
plt.show()
avg_acdnts_per_yr_in_sts = df_month.groupby('YEAR').sum()
fig = plt.figure()
avg_acdnts_per_yr_in_sts['TOTAL'].plot("bar", figsize = (12,6), title = '# of Accidents in each year')
plt.ylabel('Total # of Accidents')
plt.ylim(avg_acdnts_per_yr_in_sts['TOTAL'].min()*0.98, avg_acdnts_per_yr_in_sts['TOTAL'].max()*1.02)
plt.show()
avg_acdnts_per_yr_in_sts = df_month.groupby('YEAR').sum()
fig = plt.figure(figsize=(20,12))
ax = fig.add_subplot(111)
for col in avg_acdnts_per_yr_in_sts.drop('TOTAL',axis=1).columns:
    avg_acdnts_per_yr_in_sts[col].plot("line", ax=ax, label=col,fontsize=20)
plt.title("Monthly Total Number of Accidents")
plt.ylabel('Total # of Accidents')
plt.legend(loc='upper left', fontsize = 'xx-large')
ax.title.set_fontsize(30)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
plt.show()
avg_acdnts_per_yr_in_sts = df_month.groupby('YEAR').sum().drop('TOTAL',axis=1)
corr = np.corrcoef(avg_acdnts_per_yr_in_sts)
corr
avg_acdnts_per_yr_in_sts = df_month.groupby('TYPE').sum()
fig = plt.figure()

((100.*avg_acdnts_per_yr_in_sts['TOTAL'])/avg_acdnts_per_yr_in_sts['TOTAL'].sum()).plot("bar", figsize = (12,6), title = 'Total # of Accidents in each Type')
plt.xlabel('Accident Type')
plt.ylabel('Total # of Accidents (%)')
plt.show()
types_of_accidents = df_month['TYPE'].unique()
k = 0
colors = np.array(['Red','Green','Blue'])
for type_i in types_of_accidents:
    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111)
    acdnts_type_i = df_month[df_month['TYPE']==type_i]
    avg_acdnts_type_i = acdnts_type_i.groupby('STATE/UT').sum()/total_num_of_years
    avg_acdnts_type_i.sort_values('TOTAL',ascending=False,inplace=True)
    avg_acdnts_type_i['TOTAL'].plot("bar",  color = colors[k],width = 0.3 )
    k = k+1
    plt.title(type_i)
    plt.ylabel('Avg # of Accidents per Year')
    ax.title.set_fontsize(30)
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    plt.show()

types_of_accidents = df_time['TYPE'].unique()
time_slots = df_time.drop(['STATE/UT','YEAR','TYPE','Total'],axis=1).columns
colors = np.array(['r','g','b', 'c', 'm', 'k', 'y','#FF4567EF'])

for type_i in types_of_accidents:
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(111)
    acdnts_type_i = df_time[df_time['TYPE']==type_i]
    num_acdnts = acdnts_type_i.groupby('STATE/UT').sum().drop(['YEAR','Total'],axis=1)
    num_acdnts.plot.bar(  ax = ax ,width=1.5, fontsize=20)
    plt.title(type_i)
    plt.ylabel('Avg # of Accidents per Year')
    plt.legend(loc='upper left', fontsize = 'xx-large')
    ax.title.set_fontsize(40)
    ax.xaxis.label.set_fontsize(30)
    ax.yaxis.label.set_fontsize(20)
    plt.show()
    

