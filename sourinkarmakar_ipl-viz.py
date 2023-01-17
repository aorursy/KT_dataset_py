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
import matplotlib.pyplot as plt
matches = pd.read_csv("/kaggle/input/ipldata/matches.csv") #/kaggle/input/ipldata/deliveries.csv
matches.head()
matches = matches.replace(to_replace='Rising Pune Supergiant',value='Rising Pune Supergiants')
matches = matches.replace(to_replace='Bangalore',value='Bengaluru')
matches = matches.replace(to_replace='M Chinnaswamy Stadium',value='M. Chinnaswamy Stadium')
print(pd.unique(matches['result']))
print(pd.unique(matches['city']))
mt = matches.groupby("season")
mt_cnt = {}
for i in mt.groups.keys():
    mt_cnt[i] = len(mt.groups[i])
mt_cnt
matches['number']=[1 for i in range(len(matches))]
matches.head()
print(pd.unique(matches['id']))
mtmp = matches[['id','season','city','date','venue']]
city = pd.pivot_table(matches, index = ['city','venue'], values = 'number', aggfunc='sum')
city = city.sort_values(by='number',ascending=True)
plt.figure(figsize=(8,18))
city.plot(kind='barh',figsize=(8,12))
team1 = pd.pivot_table(matches, index = ['team1'], values = 'number', aggfunc='sum')
team2 = pd.pivot_table(matches, index = ['team2'], values = 'number', aggfunc='sum')
tteam = merged_inner = pd.merge(team1,team2,left_index=True,right_index=True)
tteam['total'] = np.array(tteam['number_x'].tolist())+np.array(tteam['number_y'].tolist())
tteam
winn = pd.pivot_table(matches, index = ['winner'], values = 'number', aggfunc='sum')
winteam = merged_inner = pd.merge(tteam,winn,left_index=True,right_index=True)
winteam
winteam[['total','number']].sort_values(by='total').plot(kind='barh',figsize=(6,9))
winteam['win_per'] = [i/j*100 for i,j in zip(winteam['number'].tolist(),winteam['total'].tolist())]
winteam
winteam[['win_per']].sort_values(by='win_per').plot(kind='barh')
toss_winner = pd.pivot_table(matches, index = ['toss_winner'], values = 'number', aggfunc='sum')
tteam['toss_win'] = toss_winner['number'].tolist()
tteam['toss_win_per'] = [j/i*100 for i,j in zip(tteam['total'].tolist(),tteam['toss_win'].tolist())]
tteam = tteam.sort_values(by='toss_win_per',ascending=True)
tteam
m1 = tteam[['toss_win','total']]
m1 = m1.sort_values(by='total')
m1.plot(kind='barh',figsize=(6,9))
plt.figure(figsize=(12,6))
plt.barh(tteam.index.tolist(),tteam['toss_win_per'].tolist())
td = pd.pivot_table(matches, index = ['toss_decision'], values = 'number', aggfunc='sum')
td
td.plot(kind='pie',subplots=True,figsize=(5,5),autopct='%1.1f%%')
run_win = matches['win_by_runs'].tolist()
run_win = [i for i in run_win if i!=0]
plt.figure(figsize=(10,5))
plt.hist(run_win,bins=50)
plt.show()
wic_win = matches['win_by_wickets'].tolist()
wic_win = [i for i in wic_win if i!=0]
plt.figure(figsize=(10,5))
plt.hist(wic_win,bins=10)
plt.show()
m2 = matches[['toss_winner','winner']]
tw = {i:0 for i in pd.unique(matches['team1'])}
print(tw)
for i in range(len(m2)):
    if m2.iloc[i]['toss_winner']==m2.iloc[i]['winner']:
        tw[m2.iloc[i]['toss_winner']]+=1
twd = pd.DataFrame(tw.values(),index=tw.keys(),columns=['number'])
twd
twd2 = merged_inner = pd.merge(twd,tteam[['toss_win']],left_index=True,right_index=True)
twd2['toss_win_to_win_per']=[i/j*100 for i,j in zip(twd2['number'].tolist(),twd2['toss_win'].tolist())]
twd2
m3 = twd2[['number','toss_win']]
m3 = m3.sort_values(by='toss_win')
m3.plot(kind='barh',figsize=(6,9))
twd2[['toss_win_to_win_per']].sort_values(by='toss_win_to_win_per').plot(kind='barh',figsize=(8,8))
pom = pd.pivot_table(matches, index = ['player_of_match'], values = 'number', aggfunc='sum').sort_values(by='number',ascending=False)[0:20]
pom
pom.sort_values(by='number').plot(kind='barh',figsize=(8,8))
ump1 = pd.pivot_table(matches, index = ['umpire1'], values = 'number', aggfunc='sum').sort_values(by='number',ascending=False)[0:10]
ump2 = pd.pivot_table(matches, index = ['umpire2'], values = 'number', aggfunc='sum').sort_values(by='number',ascending=False)[0:10]
ump3 = pd.pivot_table(matches, index = ['umpire3'], values = 'number', aggfunc='sum').sort_values(by='number',ascending=False)[0:10]
ump1,ump2,ump3
result = pd.pivot_table(matches, index = ['result'], values = 'number', aggfunc='sum').sort_values(by='number',ascending=False)
result.plot(kind='pie',subplots=True,figsize=(5,5),autopct='%1.1f%%')
data = pd.read_csv("/kaggle/input/ipldata/deliveries.csv") #
print(data.columns)
data.head()
data = data.replace(to_replace='Rising Pune Supergiant',value='Rising Pune Supergiants')
print("Total runs in IPL : ",sum(data['total_runs'].tolist()))
print("Total extra runs  : ",sum(data['extra_runs'].tolist()))
print("Total batsman runs: ",sum(data['batsman_runs'].tolist()))
most_runs = pd.pivot_table(data, index = ['batsman'], values = ['batsman_runs'], aggfunc='sum').sort_values(by='batsman_runs',ascending=False)[0:20]
most_runs
extra_con =  pd.pivot_table(data, index = ['bowler'], values = ['extra_runs'], aggfunc='sum').sort_values(by='extra_runs',ascending=False)[0:20]
extra_con
most_runs_team = pd.pivot_table(data, index = ['batting_team'], values = ['total_runs'], aggfunc='sum').sort_values(by='total_runs',ascending=False)[0:20]
most_runs_team
edata = pd.merge(mtmp,data,left_on='id',right_on='match_id')
edata
most_runs_season = pd.pivot_table(edata, index = ['season'], values = ['total_runs'], aggfunc='sum')
most_runs_season.plot()
most_runs_season = pd.pivot_table(edata, index = ['season','inning'], values = ['total_runs'], aggfunc='sum')
most_runs_season
most_runs_season = pd.pivot_table(edata, index = ['venue','city'], values = ['total_runs'], aggfunc='sum').sort_values(by='total_runs',ascending=False)[0:20]
most_runs_season
most_runs_season = pd.pivot_table(edata, index = ['batting_team','over'], values = ['total_runs'], aggfunc='sum').sort_values(by='total_runs',ascending=False)
most_runs_season
tmp1 = pd.pivot_table(edata, index = ['batting_team','over'], values = ['id'], aggfunc='count')
tmp1
twd4 = merged_inner = pd.merge(tmp1,most_runs_season,left_index=True,right_index=True)
twd4
twd4['arpo'] = [x*6/y for x,y in zip(twd4['total_runs'].tolist(),twd4['id'].tolist())]
twd4
twd4 = twd4.sort_index()
teams = list(set([x[0] for x in twd4.index]))
overs = list(set([x[1] for x in twd4.index]))
plt.figure(figsize=(18,10))
tmr={}
for i in range(len(teams)):
    runs = []
    for k in overs:
        runs.append(twd4.loc[(teams[i],k)]['arpo'])
    tmr[teams[i]]=runs
tmr
ent_crt={}
for i in tmr.keys():
    if i not in ent_crt.keys():
        ent_crt[i]=[]
    ent_crt[i].append(sum(tmr[i][0:6])/6)
    ent_crt[i].append(sum(tmr[i][6:10])/4)
    ent_crt[i].append(sum(tmr[i][10:16])/6)
    ent_crt[i].append(sum(tmr[i][16:])/4)
ent_crt
runrates = pd.DataFrame(ent_crt.values(),index=ent_crt.keys(),columns=['0-6','7-10','11-15','16-20'])
runrates.sort_values(by='16-20').plot(kind='barh',figsize=(8,12))
