import numpy as np
import pandas as pd
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
from IPython.display import Image
%matplotlib inline
pd.set_option('display.max.rows',None)
pd.set_option('display.max.columns' ,None)
df=pd.read_csv('../input/viktoraxelsen/Viktor-Axelsen.csv')
df
#Number of Tournaments Played in the year 2019

Number_of_tournaments_played=df.TOURNAMENT.unique()
print('Number of Tournaments Played in the year 2019 :',len(Number_of_tournaments_played))
#Number of Rounds Played in the year 2019

Number_of_Rounds_played=df.MATCH_DETAIL
print('Number of Rounds Played in the year 2019 :',len(Number_of_Rounds_played))
#Number of Rounds Played in each Tournament

Number_of_Rounds_Played_in_each_Tournament=df['TOURNAMENT'].value_counts()
print('Number of Rounds Played in each Tournament\n\n',Number_of_Rounds_Played_in_each_Tournament)

Number_of_Rounds_Played_in_each_Tournament.plot(kind='barh',
                                                color='#4EE2EC',
                                                figsize = (8, 8),
                                                title='Number of Rounds Played in each Tournament')
plt.show()
#Number of times he has reached round 1,2,3,semifinal,final
df['MATCH_DETAIL'].value_counts()
#Win Percentage and lost Percentage
plt.title('Win Percentage and lost Percentage',bbox={'facecolor':'0.8', 'pad':5},pad=85)
plt.pie(df['RESULT'].value_counts(),
        labels=['WIN','LOST'],
        shadow=False,
        colors=['#95DBE5FF','#EFEFE8FF'],
        radius=2,
        explode=(0.03, 0.03),
        autopct='%1.1f%%',
        startangle=90)

plt.show()
#SERVICE_ERROR with respect to opponent
C=df.loc[:,['OPPONENT','RESULT','SERVICE_ERROR']]
C.sort_values(by=['SERVICE_ERROR'], ascending=False)
plt.figure(figsize=(8, 10))
sns.scatterplot(data=C, x="SERVICE_ERROR", y="OPPONENT").set(title='Service Error with respect to opponent')
plt.show()
#STRAIGHT_TOSS_ERROR
fig, ax = plt.subplots(figsize=(10,10))
sns.swarmplot(x='STRAIGHT_TOSS_ERROR', y='OPPONENT', data=df,size=10).set(title='STRAIGHT TOSS ERROR')
plt.show()
#CROSS_TOSS__ERROR
fig, ax = plt.subplots(figsize=(10,10))
sns.swarmplot(x='CROSS_TOSS__ERROR', y='OPPONENT', data=df,size=10).set(title='CROSS TOSS ERROR')
plt.show()
#OVERALL MAXIMUM ERROR
M=df.loc[:,['SERVICE_ERROR','STRAIGHT_TOSS_ERROR','CROSS_TOSS__ERROR','STRAIGHT_SMASH_ERROR','CROSS_SMASH__ERROR','SMASH_RETURN_ERROR','STRAIGHT_DROP_ERROR','CROSS_DROP__ERROR','NET_SHOT_ERROR','NET_KILL_ERROR','NET_LIFT_ERROR']]
OVERALL_MAXIMUM_ERROR=M.sum()
OVERALL_MAXIMUM_ERROR
o=df.loc[:,['OPPONENT','SMASH_RETURN_ERROR','RESULT']]
H=o.sort_values(by=['SMASH_RETURN_ERROR'], ascending=False).head(10)
H
# Smash Return Error against Opponents
plt.figure(figsize=(8, 10))
ax = sns.barplot(x="SMASH_RETURN_ERROR", y="OPPONENT", data=H,ci=None,hue='RESULT').set(title='Smash Return Error Against Opponents')
#OVERALL LOSS RESULT
J=df.loc[:,['MATCH_DETAIL','RESULT','TOTAL_ERRORS','SERVICE_ERROR','STRAIGHT_TOSS_ERROR','CROSS_TOSS__ERROR','STRAIGHT_SMASH_ERROR','CROSS_SMASH__ERROR','SMASH_RETURN_ERROR','STRAIGHT_DROP_ERROR','CROSS_DROP__ERROR','NET_SHOT_ERROR','NET_KILL_ERROR','NET_LIFT_ERROR']]
K=J[J['RESULT']=='LOST']
K
K.plot(x='TOTAL_ERRORS', y=['SERVICE_ERROR','STRAIGHT_TOSS_ERROR',
                        'CROSS_TOSS__ERROR','STRAIGHT_SMASH_ERROR',
                        'CROSS_SMASH__ERROR','SMASH_RETURN_ERROR',
                        'STRAIGHT_DROP_ERROR','CROSS_DROP__ERROR','NET_SHOT_ERROR',
                        'NET_KILL_ERROR','NET_LIFT_ERROR'], kind='bar',figsize=(20,8),title='OVERALL LOSS RESULT (TOTAL_ERRORS)')
plt.show()

df1=df.loc[:,['OPPONENT','RESULT']]
X=df1[df1['RESULT']=='LOST']
Y=X['OPPONENT'].value_counts()
Y
df12=df.loc[:,['OPPONENT','MATCH_DETAIL','RESULT','TOTAL_ERRORS']]
Z=df12[df12['OPPONENT']=='Kento MOMOTA']
Z