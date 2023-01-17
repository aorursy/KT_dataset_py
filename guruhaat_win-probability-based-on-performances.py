# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#%%

matches=pd.read_csv("../input/matches.csv",encoding='utf-8')
ball=pd.read_csv("../input/deliveries.csv",encoding='utf-8')



#%%
def draw_chart(Score,Won,noOfMatches, title, xTickLabel='Score\n(No of Matches)'):
    fig=plt.figure(1,figsize=(8,5),facecolor="lightblue",linewidth=5,edgecolor='blue')
    plt.suptitle(title, fontsize=15)
    ax=plt.subplot(1,1,1)
    ax.plot(range(len(Won)),Won,color="red")
    ax.set_xticks(range(len(Score)))
    ax.set_xticklabels([str(x) + '\n(' +  str(y) + ')' for x,y in zip(Score,noOfMatches)])
    ax.set_xlabel(xTickLabel) 
    ax.set_ylabel('Win Probability')
    plt.grid(True)
    plt.show() 


#%%
cen=ball.groupby(['match_id','batting_team','batsman'])['batsman_runs'].sum().reset_index()
cen=cen.merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
cen['result']=cen['batting_team'] == cen['winner']
score=[]
won=[]
NoOfMatches=[]
for i in range(20,180,20):
    cenn=cen[cen['batsman_runs']>i].drop_duplicates(['match_id','batting_team']).merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
    won.append(cenn['result'].value_counts(True)[True])
    score.append(i)
    NoOfMatches.append(cenn['result'].count())
draw_chart(score,won,NoOfMatches,'Individual Score vs Win Probability')

#%%

fifty=ball.groupby(['match_id','batting_team','batsman'])['batsman_runs'].sum().reset_index()
score=[]
won=[]
NoOfMatches=[]
for i in range(40,100,10):
    fifty_result=fifty[fifty.batsman_runs>=i]
    fifty_result=fifty_result.groupby(['match_id','batting_team'])['batsman_runs'].count().reset_index()
    fifty_result=fifty_result[fifty_result['batsman_runs']>=2]
    fifty_result=fifty_result.merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
    fifty_result['result']=fifty_result['batting_team']==fifty_result['winner']
    score.append(i)
    won.append(fifty_result['result'].value_counts(True)[True])
    NoOfMatches.append(fifty_result['result'].count())
draw_chart(score,won,NoOfMatches,'Two Batsman Score vs Win Probability')


#%%

Fover=ball[ball['over']<=5].groupby(['match_id','batting_team'])['total_runs'].sum().reset_index()
Fover_result=Fover.merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
Fover_result['result']=Fover_result['batting_team']==Fover_result['winner']
score=[]
won=[]
NoOfMatches=[]
for i in range(20,90,10):
    won.append(Fover_result[Fover_result['total_runs']>i]['result'].value_counts(True)[True])
    score.append(i)
    NoOfMatches.append(Fover_result[Fover_result['total_runs']>i]['result'].count())
draw_chart(score,won,NoOfMatches,'First 5 Overs Score vs Win Probability')    

    
#%%

Fover=ball[ball['over']>=16].groupby(['match_id','batting_team'])['total_runs'].sum().reset_index()
Fover_result=Fover.merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
Fover_result['result']=Fover_result['batting_team']==Fover_result['winner']
score=[]
won=[]
NoOfMatches=[]
for i in range(20,120,10):
    won.append(Fover_result[Fover_result['total_runs']>i]['result'].value_counts(True)[True])
    score.append(i)
    NoOfMatches.append(Fover_result[Fover_result['total_runs']>i]['result'].count())
draw_chart(score,won,NoOfMatches,'Last 5 Overs Score vs Win Probability')    

#%%
ball['wicket']=ball['player_dismissed'].apply(lambda x: 1 if x is not np.nan else 0).cumsum()
patnership=ball.groupby(['match_id','batting_team','wicket'])['total_runs'].sum().reset_index().sort_values(['wicket','batting_team'])
score=[]
won=[]
NoOfMatches=[]
for i in range(40,200,20):
    patnership_result=patnership[patnership['total_runs']>i]
    patnership_result=patnership_result.drop_duplicates(['match_id','batting_team']).merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
    patnership_result['result']=patnership_result['batting_team']==patnership_result['winner']
    won.append(patnership_result['result'].value_counts(True)[True])
    score.append(i)
    NoOfMatches.append(patnership_result['result'].count())
draw_chart(score,won,NoOfMatches,'Parternship vs Win Probability')


#%%
opening_partnership=patnership.sort_values('wicket').drop_duplicates(subset=['match_id','batting_team'],keep='first')
score=[]
won=[]
NoOfMatches=[]
for i in range(20,180,20):
    opening_patnership_result=opening_partnership[opening_partnership['total_runs']>i]
    opening_patnership_result=opening_patnership_result.drop_duplicates(['match_id','batting_team']).merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
    opening_patnership_result['result']=opening_patnership_result['batting_team']==opening_patnership_result['winner']
    won.append(opening_patnership_result['result'].value_counts(True)[True])
    score.append(i)
    NoOfMatches.append(opening_patnership_result['result'].count())
draw_chart(score,won,NoOfMatches,'Opening Parternship vs Win Probability')


#%%

score=[]
won=[]
NoOfMatches=[]
for i in range(10,70,10):
    multi_patnership=patnership[patnership['total_runs']>i].groupby(['match_id','batting_team'])['total_runs'].count().reset_index()
    multi_patnership=multi_patnership[multi_patnership['total_runs']>2].drop_duplicates(['match_id','batting_team']).merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
    multi_patnership['result']=multi_patnership['batting_team']==multi_patnership['winner']
    won.append(multi_patnership['result'].value_counts(True)[True])
    score.append(i)
    NoOfMatches.append(multi_patnership['result'].count())
draw_chart(score,won,NoOfMatches,'Multi Parternship vs Win Probability')

#%%

ball['w']=ball['player_dismissed'].apply(lambda x: 1 if x is not np.nan else 0)
#ball['ws']=ball.groupby(['match_id','batting_team'])['w'].cumsum()
wicket=ball[ball['over']<=5].groupby(['match_id','batting_team'])['w'].sum().reset_index()
score=[]
won=[]
NoOfMatches=[]
for i in range(0,7,1):
    wicket_result=wicket[wicket['w']>=i]
    wicket_result=wicket_result.merge(matches[['id','winner']],how='left',left_on='match_id',right_on='id')
    wicket_result['result']=wicket_result['batting_team']==wicket_result['winner']
    won.append(wicket_result['result'].value_counts(True)[False])
    score.append(i)
    NoOfMatches.append(wicket_result['result'].count())
draw_chart(score,won,NoOfMatches,'FIrst 5 Overs Wickets vs Win Probability','Wickets\n(No of Matches)')

