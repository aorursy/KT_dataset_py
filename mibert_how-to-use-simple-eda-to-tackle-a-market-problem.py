import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import matplotlib.cm as cm
import seaborn as sns

%matplotlib inline
plt.style.use('seaborn-white')
# Data Set for gaming hours of steam

st = pd.read_csv('../input/steam-video-games/steam-200k.csv',names=['User_ID','Game','Action','Value','Other'])
st.head(10)
# Data Set for Metacritic Ratings

rt = pd.read_csv('../input/metacritic-games-stats-20112019/metacritic_games.csv')
rt.head(10)
#Removing User ID's as they are of no interest in this EDA 

st = st[['Game','Action','Value']]

#Taking only play hours of the games and dropping the absolete action column

st = st[st['Action'] == 'play']
st.drop(['Action'],inplace=True,axis=1)
st.rename(columns={'Value':'Hours Played'},inplace=True)

#Taking Cumulative Playing time of all the users

st = st.groupby('Game').sum().reset_index()
st = st.sort_values('Hours Played',ascending = False).reset_index(drop=True)

#top 10 Most Played Games

st.head(10)
#Keeping only relevant columns

col = rt.columns
rt = rt[col[:2].tolist()+col[7:-2].tolist()]

#Taking only PC games
rt = rt[rt['platform'] == 'PC']
rt.drop(['platform'],axis=1,inplace=True)

rt.head()
#Score

rt['neutral_critics'] = rt['neutral_critics']*0.5
rt['negative_critics'] = rt['negative_critics']*(-1)
rt['neutral_users'] = rt['neutral_users']*0.5
rt['negative_users'] = rt['negative_users']*(-1)
rt['Score'] = rt['positive_critics'] + rt['neutral_critics'] + rt['negative_critics'] + rt['neutral_users'] + rt['negative_users'] + rt['positive_users']
rt = rt[['game','Score']].rename(columns={'game':'Game'})
rt = rt.sort_values('Score',ascending=False).reset_index(drop=True)

#Top 10 Rated Games
rt.head(10)
# Developing the merge Datasets

final = pd.merge(st,rt,how='inner',left_on='Game',right_on='Game')
topst = final.sort_values('Hours Played',ascending=False)[2:12]
topst
toprt = final.sort_values('Score',ascending=False)[2:12]
toprt
#Function to normalize the values

def Normalize(lst):
    norm = []
    mx = max(lst)
    mn = min(lst)
    for i in lst:
        norm.append( ((i - mn) / (mx - mn)) * 2 )
    return norm
# Normalizing the Scores

topst['Play Score'] = Normalize(topst['Hours Played'].tolist())
topst['Rating Score'] = Normalize(topst['Score'].tolist())
topst.reset_index(drop=True,inplace=True)
topst
# Normalizing the Scores

toprt['Play Score'] = Normalize(toprt['Hours Played'].tolist())
toprt['Rating Score'] = Normalize(toprt['Score'].tolist())
toprt.reset_index(drop=False,inplace=True)
toprt
#testing for trends in most played games

fig1 = plt.figure(figsize=(8,6.5))
plt.plot(topst['Play Score'],'-o',label='Play Score',c='orange')
plt.plot(topst['Rating Score'],'-o',label='Rating Score',c='b')
plt.legend(title='Legend');
spines1 = plt.gca().spines
spines1['right'].set_visible(False)
spines1['top'].set_visible(False)
spines1['left'].set_visible(False)
spines1['bottom'].set_visible(False)
plt.grid()
plt.title('Play Time of Most Played games V/S Rating Score.');
plt.xlabel('Top Game Ranks');
plt.ylabel('Normalized Score Scale');
#testing for trends in top rated games

fig2 = plt.figure(figsize=(8.,6.5))
plt.plot(toprt['Play Score'],'-o',label='Play Score',c='orange')
plt.plot(toprt['Rating Score'],'-o',label='Rating Score',c='b')
plt.legend()
spines2 = plt.gca().spines
spines2['right'].set_visible(False)
spines2['top'].set_visible(False)
spines2['left'].set_visible(False)
spines2['bottom'].set_visible(False)
plt.title('Play Time of Top Rated games V/S Rating Score.');
plt.xlabel('Top Game Ranks');
plt.ylabel('Normalized Score Scale');
plt.legend(title='Legend')
plt.grid()
canv,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
canv.set_size_inches(18,15)
canv.tight_layout(pad=5.0)

#First Plot,(0,0)

plt.sca(ax1)
plt.rcParams.update({'font.size': 14})
bars = plt.barh(np.arange(len(topst['Game'])),topst['Hours Played'].iloc[::-1],color='lightslategrey',alpha=0.7)
bars[-1].set_color('orange')

for bar,name,value in zip(bars,topst['Game'].iloc[::-1].tolist(),topst['Hours Played'].iloc[::-1].tolist()):
    plt.text((bar.get_width()/4)-2500,(bar.get_y()+0.3),name + ' ({:.0f} Hours)'.format(value),color='w',fontweight='bold',fontsize=13)

plt.yticks(np.arange(len(topst['Game'])),np.array([10,9,8,7,6,5,4,3,2,1]));
ax1.set_xticks([])
plt.xlabel('Hours Played on Steam.',fontsize=15)
plt.ylabel('Ranking Based on Play Time on Steam.',fontsize=15)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

#Second Plot,(0,1)

plt.sca(ax2)
plt.rcParams.update({'font.size': 14})
plt.plot(topst['Play Score'],'-o',label='Play Score',c='orange')
plt.plot(topst['Rating Score'],'-o',label='Rating Score',c='b')
plt.legend(title='Legend');

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.grid(alpha=0.8)
plt.title('Play Time of Most Played games V/S Rating Score.');
plt.xticks(np.arange(len(topst['Game'])),np.array([1,2,3,4,5,6,7,8,9,10]));
plt.xlabel('Ranking Based on Play Time on Steam.',fontsize=15);
plt.ylabel('Normalized Score Scale.',fontsize=15);

#Third Plot,(1,0)

plt.sca(ax3)
plt.rcParams.update({'font.size': 14})
bars = plt.barh(np.arange(len(toprt['Game'])),toprt['Score'].iloc[::-1],color='lightslategrey',alpha=0.7)
bars[-1].set_color('b')

for bar,name,value in zip(bars,toprt['Game'].iloc[::-1].tolist(),toprt['Score'].iloc[::-1].tolist()):
    plt.text((bar.get_width()/4-10),(bar.get_y()+0.3),name + ' ({:.0f})'.format(value),color='w',fontweight='bold',fontsize=13)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.yticks(np.arange(len(toprt['Game'])),np.array([10,9,8,7,6,5,4,3,2,1]));
ax3.set_xticks([])
plt.xlabel('Rating Score on Steam.',fontsize=15)
plt.ylabel('Ranking Based on Rating Score on Steam.',fontsize=15)

#Fourth Plot(1,1)

plt.sca(ax4)
plt.rcParams.update({'font.size': 14})
plt.plot(toprt['Play Score'],'-o',label='Play Score',c='orange')
plt.plot(toprt['Rating Score'],'-o',label='Rating Score',c='b')
plt.legend()

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.title('Play Time of Top Rated games V/S Rating Score.');
plt.xlabel('Ranking Based on Rating Score on Steam.',fontsize=15);
plt.xticks(np.arange(len(toprt['Game'])),np.array([1,2,3,4,5,6,7,8,9,10]));
plt.ylabel('Normalized Score Scale.',fontsize=15);
plt.legend(title='Legend');
plt.grid(alpha=0.8)