# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df1=pd.read_csv('../input/game_skater_stats.csv')
df2=pd.read_csv('../input/player_info.csv')
df3=pd.merge(df1,df2,how='left',on='player_id',left_index=True,right_index=True)
df1.head()
df2.head()
df3.head()
sum1=[]
for group,frame in df3.groupby('team_id'):
    sum1.append(frame['goals'].sum())
s=df3.sort_values(by='team_id')
df=pd.DataFrame(data=sum1,index=s['team_id'].unique())
df.iplot(kind='bar',xTitle='Team Id',yTitle='No of Goals')
plt.figure()
sum2=[]
position=[]
for group,frame in df3.groupby('primaryPosition'):
    sum2.append(frame['goals'].sum())
    position.append(group)
    
sns.barplot(x=position,y=sum2)    
plt.xlabel('Positions')
plt.ylabel('No of Goals')
plt.title('Goals Scored per Position')   
plt.figure()
sum3=[]
countries=[]
for groups,frame in df3.groupby('nationality'):
    sum3.append(frame['goals'].sum())
    countries.append(groups)
    
dfz=pd.DataFrame(sum3,index=countries)
dfz.iplot(kind='bar',xTitle='Countries',yTitle='Goals',title='Goals per Country')
    

    
df4=pd.read_csv('../input/game.csv')
df4.head()
df5=pd.merge(df4,df1,how='left',on='game_id',left_index=True,right_index=True)
df5.head()
away=[]
home=[]
team=[]
for groups,frame in df5.groupby('team_id'):
    away.append(frame['away_goals'].sum())
    home.append(frame['home_goals'].sum())
    team.append(groups)
plt.figure()
dfz1=pd.DataFrame(data=away,index=team)
dfz2=pd.DataFrame(data=home,index=team)
dfz3=pd.merge(dfz1,dfz2,how='left',left_index=True,right_index=True)
#dfz3[['0_x','0_y']].plot(kind='hist')

#plt.subplot(1,2,1)
plt.tight_layout()
sns.barplot(x=dfz3.index,y=dfz3['0_x'])
plt.xlabel('Team Id')
plt.ylabel('Away Goals')
#plt.subplot(1,2,2)
#sns.barplot(x=df3.index,y=dfz3['0_y'])
#plt.xlabel('Team Id')
#plt.ylabel('Home Goals')

sns.barplot(x=dfz3.index,y=dfz3['0_y'])
plt.xlabel('Team Id')
plt.ylabel('Home Goals')



