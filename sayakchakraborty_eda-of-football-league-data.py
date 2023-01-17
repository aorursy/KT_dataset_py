local_path="../input/"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
%matplotlib inline
col=['Date','HomeTeam','AwayTeam','Referee','FTHG','FTAG','FTR','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR']
df_2012_13=pd.read_csv(local_path+'2012-13.csv',usecols=col,index_col='Date')
df_2012_13.tail()
df_merge=df_2012_13.append(pd.read_csv(local_path+'2013-14.csv',usecols=col,index_col='Date'),ignore_index=False)
df_merge=df_merge.append(pd.read_csv(local_path+'2014-15.csv',usecols=col,index_col='Date'),ignore_index=False)
df_merge=df_merge.append(pd.read_csv(local_path+'2015-16.csv',usecols=col,index_col='Date'),ignore_index=False)

df_merge.shape
df_merge.dropna(axis=0,how='all',inplace=True)
len(df_merge)-df_merge.count()
df_merge.describe()
from collections import Counter
htmc=Counter(df_merge.HomeTeam)
print('Home Team Match:',htmc)
df_merge.groupby('HomeTeam').count()['FTR'].sort_values(ascending=False)
df_merge.groupby('AwayTeam').count()['FTR'].sort_values(ascending=False)
df_merge['Winner']='NA'
df_merge.head()
df_merge['Winner'][df_merge['FTR']=='H']=df_merge['HomeTeam']
df_merge['Winner'][df_merge['FTR']=='A']=df_merge['AwayTeam']
df_merge.head()
x=100*df_merge.groupby('FTR').count()['HomeTeam']/len(df_merge)
colors = ['lightgreen', 'lightcoral', 'lightskyblue']
plt.pie(x,explode=[.05,0,.05],labels=['AwayTeam Win%='+str(x.ix[0].round(1)),
                                    'Draw%='+str(x.ix[1].round(1)),'HomeTeam Win%='+str(x.ix[2].round(1))],
        startangle=90,shadow=True,colors=colors,textprops={'Fontsize':'large', 'fontweight':'bold'})
plt.axis('equal')
plt.title('Win Percentage Home and Away',fontsize=14,fontweight='bold')
plt.figure(figsize=(18,10))
plt.subplots_adjust(hspace=0.2,wspace=0.4)
plt.suptitle("Matches Won by Team Insight",fontsize=18,fontweight='bold')
                    
plt.subplot(1,3,1)
df_merge.groupby('Winner').count()['FTR'].sort_values(ascending=True).ix[1:-1].plot(kind='barh',grid=True)
plt.axvline(np.mean(df_merge.groupby('Winner').count()['FTR']),color='r')
#df_win.ix[1:-1]
plt.title('All season combined winners')
plt.xlabel('Count of matches')
#plt.tight_layout()

plt.subplot(1,3,2)
df_home_team=df_merge.pivot_table(values='AwayTeam',index='HomeTeam',columns='FTR',aggfunc='count')
Total=df_home_team.sum(axis=1)
for i in df_home_team.columns.tolist():
    df_home_team['Percent:'+i]=100*df_home_team[i]/Total
sns.heatmap(df_home_team[['Percent:A','Percent:D','Percent:H']],cmap='coolwarm',annot=True,cbar=False)
plt.title('Percentage of Win/Loss/Draw at Home Ground')
#plt.tight_layout()

plt.subplot(1,3,3)
df_away_team=df_merge.pivot_table(values='HomeTeam',index='AwayTeam',columns='FTR',aggfunc='count')
Total=df_away_team.sum(axis=1)
for i in df_away_team.columns.tolist():
    df_away_team['Percent:'+i]=100*df_away_team[i]/Total
sns.heatmap(df_away_team[['Percent:A','Percent:D','Percent:H']],cmap='coolwarm',annot=True)
plt.title('Percentage of Win/Loss/Draw at Away Ground')
#plt.tight_layout()
                    
#df_merge
###This portion is not working
'''
for i in range(0,len(df_merge)):
    if df_merge.iloc[i]['FTR']=='H':
        df_merge.iloc[i]['Winner']=df_merge.iloc[i]['HomeTeam']
    elif df_merge.iloc[i]['FTR']=='A':
        df_merge.iloc[i]['Winner']=df_merge.iloc[i]['AwayTeam']
    else:
        df_merge.iloc[i]['Winner']='Drawn'
df_merge.head()
'''
#df_merge.head()
df_merge.iloc[1]['FTR']
pd.set_option('display.max_columns',500)
df_goal=df_merge.pivot_table(values=['FTHG','FTAG'],index='HomeTeam',columns='AwayTeam',aggfunc='sum')
#df_goal
plt.figure(figsize=(10,10))
plt.subplots_adjust(wspace= .6)
plt.suptitle('Total goals made by teams',fontsize=18)
plt.subplot(1,2,1)
df_goal.xs('FTHG',level=0,axis=1).sum(axis=1).sort_values(ascending=True).plot(kind='barh')
plt.title('Total goal at Home Ground')
plt.subplot(1,2,2)
df_goal.xs('FTAG',level=0,axis=1).sum(axis=0).sort_values().plot(kind='barh',color='g')
plt.title('Total goal at Away Ground')
plt.figure(figsize=(20,10))
plt.subplots_adjust(wspace=0.1)
plt.suptitle('Goal made by team against each team',fontsize=20)
plt.subplot(1,2,1)
sns.heatmap(df_goal.xs('FTHG',level=0,axis=1),cmap='coolwarm',annot=True)
plt.title('Score of Teams at Home Ground', fontsize=16)
#sns.heatmap(df_HGgoal.xs)
plt.subplot(1,2,2)
sns.heatmap(df_goal.xs('FTAG',level=0,axis=1),cmap='Blues',annot=True)
plt.title('Score of Teams at Away Ground',fontsize=16)
df_win=df_merge.groupby('Winner').count()['FTR']
df_total_goals=df_goal.xs('FTHG',level=0,axis=1).sum(axis=1)+df_goal.xs('FTAG',level=0,axis=1).sum(axis=0)
df_win_goal=pd.concat({'Count_Win':df_win,'Total_goal':df_total_goals},axis=1).dropna()
#plt.scatter(df_win_goal.Count_Win,df_win_goal.Total_goal)
sns.regplot('Count_Win','Total_goal',df_win_goal)
plt.title('Corelation of Win vs Score',fontsize=16)
df_merge.head()
df_merge_piv_h=df_merge.pivot_table(values=['FTHG','HST','HF','HC','HY','HR'],index='HomeTeam',aggfunc=np.mean)
df_merge_piv_h['Team']='HomeTeam'
df_merge_piv_h.rename(index=str,columns={'FTHG':'Goal','HC':'Corner','HF':'Foul','HR':'Red_Card','HST':'Shot_Target',
                                      'HY':'Yellow_Card'},inplace=True)
df_merge_piv_h
df_merge_piv_a=df_merge.pivot_table(values=['FTAG','AST','AF','AC','AY','AR'],index='AwayTeam',aggfunc=np.mean)
df_merge_piv_a['Team']='AwayTeam'
df_merge_piv_a.rename(index=str,columns={'FTAG':'Goal','AC':'Corner','AF':'Foul','AR':'Red_Card','AST':'Shot_Target',
                                        'AY':'Yellow_Card'},inplace=True)
df_m_ha=pd.concat([df_merge_piv_h,df_merge_piv_a],axis=0,ignore_index=False)
#df_m_ha.sort_values('Corner',ascending=False)
sns.pairplot(df_m_ha,hue='Team',palette='husl',kind='scatter')
plt.show()
df_m_ha.drop('Team',axis=1)
df_m_ha_m=df_m_ha.groupby(df_m_ha.index).mean()
df_m_ha_m.sort_values('Goal',ascending=False,inplace=True)
df_m_ha_m[['Corner','Shot_Target','Goal']].plot(kind='bar',figsize=(20,8),label=df_m_ha_m.index)
plt.xlabel("Team Name")
a=np.arange(len(df_m_ha_m))
plt.xticks(a,(df_m_ha_m.index),rotation=90)
plt.title('Average Goal/Shot_Target/Corner per Match', fontsize=18)
plt.show()

df_merge.groupby('Referee').count()['FTR'].sort_values()
df_Referee=df_merge.pivot_table(values=['AR','AY','HR','HY'],index='Referee',aggfunc=np.mean)
#df_Referee
fig, ax=plt.subplots(figsize=(28,12))
ax.plot()
sns.barplot(x=df_Referee.index,y=df_Referee.HY.sort_values(ascending=False),color='Yellow')
sns.barplot(x=df_Referee.index,y=-df_Referee.AY.sort_values(ascending=False),color='Yellow')
plt.ylabel('Avg Yellow Cards\n Away Team--------------------------Home Team')
ax2=ax.twinx()
sns.barplot(x=df_Referee.index,y=df_Referee.HR.sort_values(ascending=False),color='Red',facecolor=(1, 1, 1, 0),linewidth=2.5,edgecolor='Red')
sns.barplot(x=df_Referee.index,y=-df_Referee.AR.sort_values(ascending=False),color='Red',facecolor=(1, 1, 1, 0),linewidth=2.5,edgecolor="Red")
plt.axhline(0,color='Black')
plt.ylabel('Avg Red Cards\n Away Team--------------------------Home Team')
plt.title('Average Yellow and Red Card per Referee', fontsize=18)
plt.show()
#df_2012_13.tail()
df_season1=df_merge.ix[:'19/05/2013'].groupby('HomeTeam').count()['Referee']
df_season2=df_merge.ix['17/08/2013':'11/05/2014'].groupby('HomeTeam').count()['HST']
df_season3=df_merge.ix['16/08/2014':'24/05/2015'].groupby('HomeTeam').count()['AST']
df_season4=df_merge.ix['08/08/2015':].groupby('HomeTeam').count()['HF']
df_relg=pd.concat([df_season1,df_season2,df_season3,df_season4],axis=1)
df_relg.rename(index=str, columns={'Referee':'FY_12_13','HST':'FY_13_14','AST':'FY_14_15','HF':'FY_15_16'},inplace=True)
df_relg.fillna(0,inplace=True)
plt.figure(figsize=(8,8))
sns.heatmap(df_relg,linecolor='white',linewidths=1,center=0,cbar=False)
plt.title('Relegation and Promotion Analysis',fontsize=14)
plt.figure(figsize=(20,8))
fig, ax=plt.subplots(figsize=(28,12))
ax.plot()
sns.barplot(x=df_m_ha_m.index,y=df_m_ha_m.Foul.sort_values(ascending=False),color='lightskyblue')
sns.barplot(x=df_m_ha_m.index,y=df_m_ha_m.Yellow_Card,color='yellow')
b=np.arange(len(df_m_ha_m))
plt.xticks(b, df_m_ha_m.index,rotation=90,fontsize=14)
ax2=ax.twinx()
#sns.barplot(x=df_m_ha_m.index,y=-df_m_ha_m.Yellow_Card,color='yellow')
sns.barplot(x=df_m_ha_m.index,y=df_m_ha_m.Red_Card,color='Red',facecolor=(1, 1, 1, 0),linewidth=2.5,edgecolor="Red")
plt.title("Average Foul/Card per Match",fontsize=22,fontweight='bold')
plt.show()
df_foul=df_merge.pivot_table(values=['HF','AF'],index='HomeTeam',columns='AwayTeam',aggfunc=np.mean)
#sns.heatmap(df_foul.xs('HF'))
plt.figure(figsize=(16,8))
plt.suptitle('Fouls made by Teams',fontsize=16,fontweight='bold')
plt.subplots_adjust(hspace=.5,wspace=0.3)
plt.subplot(1,2,1)
sns.heatmap(df_foul.xs(key='HF',level=0,axis=1),annot=True,cbar=False)
plt.title('HomeTeam Fouls',fontsize=14)
plt.subplot(1,2,2)
sns.heatmap(df_foul.xs(key='AF',level=0,axis=1).T,annot=True,cbar=False)
plt.title('AwayTeam Fouls',fontsize=14)
plt.show()
