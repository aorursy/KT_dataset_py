sky_stats_by_year.iloc[1:5]
sky_stats_by_year3
colors = ['purple','blue']
pie = sky_final_runs_for_team.plot.pie(y='Runs', figsize=(12,8), colors = colors, autopct='%1.2f%%', textprops={'color':"w", 'size':'20'})
label = ['Kolkata Knight Riders (608 Runs)','Mumbai Indians (936 Runs)']
plt.legend(labels = label, loc = 'upper left')
new_data_bat_posss
x = sky_runs_bat_pos['Bat Pos']
y = sky_runs_bat_pos['Runs']
plt.figure(figsize=(12,6))
sns.barplot(x,y)
plt.xlabel('Batting Position')
plt.ylabel('Runs scored')
plt.title('Runs scored by Suryakumar at each batting position')
teamswise = sky_runs_by_oppn.join(sky_final_boundaries_by_oppn)
teamswise = teamswise.drop(columns='Opposition')
teamsswise = sky_final_sixes_by_oppn.drop(columns='Oppn')
teamswise = teamswise.join(teamsswise)
teamswise = teamswise.rename(columns={"Oppn": "Opposition", "6s": "Sixes"})
teamswise             
sky_final_boundaries_by_oppn = pd.DataFrame(sky_final_boundaries_by_oppn)
sky_final_boundaries_by_oppn.columns = ['Opposition','Fours']
x=sky_final_boundaries_by_oppn['Opposition']
y=sky_final_boundaries_by_oppn['Fours']
plt.figure(figsize=(12,6))
sns.barplot(x,y)
plt.title('Suryakumar Yadav fours against each team')
xxx = sky_final_sixes_by_oppn['Oppn']
yyy= sky_final_sixes_by_oppn['6s']
plt.figure(figsize=(12,6))
sns.barplot(xxx,yyy)
plt.title('Suryakumar sixes against each team')
plt.xlabel('Opposition')
plt.ylabel('Sixes')
colors2 = ['#ff9999','#99ff99']
pie2 = sky_runs_at_final.plot.pie(y='Runs', figsize=(12,8), colors = colors2, autopct='%1.1f%%', textprops={'size':'20'})
labell = ['Away (839 Runs)','Home (705 Runs)']
plt.legend(labels = labell, loc = 'upper right')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
sky_data = pd.read_csv('../input/suryakumar/Suryakumar.csv',encoding = 'unicode_escape')
sky_data
sky_opposition = sky_data.groupby('Oppn')
sky_opposition['Oppn'].value_counts()
sky_by_oppn = sky_data.sort_values('Oppn')
sky_by_oppn
sky_initial_runs_by_oppn = sky_by_oppn.groupby('Oppn')
sky_runs_by_oppn = sky_initial_runs_by_oppn['Runs'].sum().reset_index()
sky_runs_by_oppn
x=sky_runs_by_oppn['Oppn']
y=sky_runs_by_oppn['Runs']
plt.figure(figsize=(12,6))
sns.barplot(x,y)
plt.title('Suryakumar Yadav runs against each team')
sky_first_boundaries_by_oppn = sky_by_oppn.groupby('Oppn')
sky_final_boundaries_by_oppn = sky_first_boundaries_by_oppn['4s'].sum().reset_index()
sky_final_boundaries_by_oppn
sky_final_boundaries_by_oppn = pd.DataFrame(sky_final_boundaries_by_oppn)
sky_final_boundaries_by_oppn.columns = ['Opposition','Fours']
x=sky_final_boundaries_by_oppn['Opposition']
y=sky_final_boundaries_by_oppn['Fours']
plt.figure(figsize=(12,6))
sns.barplot(x,y)
plt.title('Suryakumar Yadav fours against each team')
sky_first_sixes_by_oppn = sky_by_oppn.groupby('Oppn')
sky_final_sixes_by_oppn = sky_first_sixes_by_oppn['6s'].sum().reset_index()
sky_final_sixes_by_oppn
xxx = sky_final_sixes_by_oppn['Oppn']
yyy= sky_final_sixes_by_oppn['6s']
plt.figure(figsize=(12,6))
sns.barplot(xxx,yyy)
plt.title('Suryakumar sixes against each team')
plt.xlabel('Opposition')
plt.ylabel('Sixes')
sky_initial_runs_for_team = sky_by_oppn.groupby('For')
sky_final_runs_for_team = sky_initial_runs_for_team['Runs'].sum()
sky_final_runs_for_team
colors = ['purple','blue']
pie = sky_final_runs_for_team.plot.pie(y='Runs', figsize=(12,8), colors = colors, autopct='%1.2f%%', textprops={'color':"w", 'size':'20'})
label = ['Kolkata Knight Riders (608 Runs)','Mumbai Indians (936 Runs)']
plt.legend(labels = label, loc = 'upper left')
sky_by_bat_pos = sky_data.sort_values('Bat Pos')
sky_by_bat_pos
sky_initial_runs_bat_pos = sky_by_oppn.groupby('Bat Pos')
sky_runs_bat_pos = sky_initial_runs_bat_pos['Runs','Balls','NO'].sum().reset_index()
sky_runs_bat_pos
xy = sky_by_oppn['Bat Pos'].value_counts().reset_index()
xy = pd.DataFrame(xy)
xy.columns = ['Bat Pos','Innings']
xyz = xy.sort_values(by=['Bat Pos']).reset_index()
xyza = xyz.drop(columns='index')
skysky = sky_runs_bat_pos.drop(columns = 'Bat Pos')
xyza = xyza.join(skysky)
AVG1 = xyza['Runs']/(xyza['Innings'] - xyza['NO'])
AVG = round(AVG1, 2)
avgxyz = AVG
avgxyz = pd.DataFrame(avgxyz)
avgxyz.columns = ['Avg.']
avgxyz
new_data = xyza.join(avgxyz)
new_data

SRbatposinitial = new_data['Runs']/new_data['Balls']*100
SRbatpos = round(SRbatposinitial,2)
SRbatpos = pd.DataFrame(SRbatpos)
SRbatpos.columns = ['SR']
SRbatpos
new_data_bat_pos = new_data.join(SRbatpos)
new_data_bat_pos
thirtiesbatpos = ['8','6','2','1','0','3','0']
fiftiesbatpos = ['4','3','0','0','0','0','0']
HSbatpos = ['72','71*','46*','31','23','34','9*']
foursbatpos = ['53','47','23','6','6','23','2']
sixesbatpos = ['14','10','12','2','1','8','0']
new_data_bat_poss = new_data_bat_pos
new_data_bat_poss['30s'] = thirtiesbatpos
new_data_bat_poss['50s'] = fiftiesbatpos
new_data_bat_poss['HS'] = HSbatpos
new_data_bat_poss['4s'] = foursbatpos
new_data_bat_poss['6s'] = sixesbatpos
new_data_bat_posss = new_data_bat_poss.rename(columns={'Bat Pos':'Position'})
new_data_bat_posss = new_data_bat_posss[['Position','Innings','Runs','Balls','NO','HS','Avg.','SR','30s','50s','4s','6s']]
new_data_bat_posss
x = sky_runs_bat_pos['Bat Pos']
y = sky_runs_bat_pos['Runs']
plt.figure(figsize=(12,6))
sns.barplot(x,y)
plt.xlabel('Batting Position')
plt.ylabel('Runs scored')
plt.title('Runs scored by Suryakumar at each batting position')
sky_runs_at = sky_data.groupby('At')
sky_runs_at_final = sky_runs_at['Runs'].sum()
sky_runs_at_final
colors2 = ['#ff9999','#99ff99']
pie2 = sky_runs_at_final.plot.pie(y='Runs', figsize=(12,8), colors = colors2, autopct='%1.1f%%', textprops={'size':'20'})
labell = ['Away (839 Runs)','Home (705 Runs)']
plt.legend(labels = labell, loc = 'upper right')
sky_runs_inn_num_initial = sky_data.groupby('Inn')
sky_runs_inn_num = sky_runs_inn_num_initial['Runs','Balls','NO'].sum().reset_index()
sky_runs_inn_num
colors2 = ['yellow','red']
pie3 = sky_runs_inn_num.plot.pie(y='Runs', figsize=(12,8), colors = colors2, autopct='%1.1f%%', textprops={'size':'20'})
labels2 = ['First innings (829 Runs)','Second innings (539 Runs)']
#plt.title('Runs scored by Suryakumar in each innings of a match')
plt.legend(labels = labels2, loc = 'upper right')
from matplotlib.pyplot import plot
SR1 = sky_runs_inn_num['Runs']/sky_runs_inn_num['Balls']*100
SR = round(SR1,2)
yy = SR
xx = sky_runs_inn_num['Inn']
sns.barplot(xx,yy)
plt.title('Strike rate in each innings of a match')
sky_year = sky_data.groupby('Year')
sky_year['Year'].value_counts()
sky_initial_stats_by_year = sky_data.groupby('Year')
sky_stats_by_year = sky_initial_stats_by_year['Runs','Balls','NO','4s','6s'].sum().reset_index()
sky_stats_by_year
SRyear = sky_stats_by_year['Runs']/sky_stats_by_year['Balls']*100
SRyearfinal = round(SRyear,2)
SRyearfinal = pd.DataFrame(SRyearfinal)
SRyearfinal.columns = ['SR']
sky_stats_by_year = sky_stats_by_year.join(SRyearfinal)
sky_stats_by_year = sky_stats_by_year[['Year','Runs','Balls','NO','SR','4s','6s']]
sky_stats_by_year
qwe = sky_data['Year'].value_counts().reset_index()
qwe = pd.DataFrame(qwe)
qwe.columns = ['Year','Innings']
qwerty = qwe.sort_values(by=['Year']).reset_index()
qwertyasd = qwerty.drop(columns='index')
qwertyasd = qwertyasd.drop(columns = 'Year')
qwertyasd
sky_stats_by_year = sky_stats_by_year.join(qwertyasd)
sky_stats_by_year = sky_stats_by_year[['Year','Innings','Runs','Balls','NO','SR','4s','6s']]
sky_stats_by_year
AVGyear = sky_stats_by_year['Runs']/(sky_stats_by_year['Innings'] - sky_stats_by_year['NO'])
AVGyearfinal = round(AVGyear,2)
AVGyearfinal = pd.DataFrame(AVGyearfinal)
AVGyearfinal.columns = ['Avg.']
AVGyearfinal
sky_stats_by_year = sky_stats_by_year.join(AVGyearfinal)
sky_stats_by_year = sky_stats_by_year[['Year','Innings','Runs','Balls','NO','Avg.','SR','4s','6s']]
sky_stats_by_year
sky30sinitial = sky_data.loc[(sky_data['Runs'] >= 30)].reset_index()
sky30sinitial = sky30sinitial.drop(columns = 'index')
sky30sinitial = sky30sinitial.groupby('Year')
sky30sinitial['Year'].value_counts()
sky_stats_by_year
thirtiesyear = ['0','2','1','1','2','9','5']
fiftiesyear = ['0','0','0','1','0','4','2']
sky_stats_by_year['30s'] = thirtiesyear
sky_stats_by_year['50s'] = fiftiesyear
sky_stats_by_year
sky_stats_by_year.iloc[1:5]
sky_stats_by_year2 = sky_stats_by_year[(sky_stats_by_year.Year <= 2013) | (sky_stats_by_year.Year >= 2018)].reset_index()
sky_stats_by_year3 = sky_stats_by_year2.drop(columns = 'index')
sky_stats_by_year3
