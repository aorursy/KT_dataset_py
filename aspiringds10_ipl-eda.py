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
import matplotlib.patches as mpatches
%matplotlib inline 
import seaborn as sns
import plotly.io as pio
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
matches = pd.read_csv('/kaggle/input/ipldata/matches.csv')
deliveries = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')
matches.head()
matches.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant','Delhi Capitals']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PWI','RPS','DCAP'],inplace=True)

deliveries.replace(['Mumbai Indians','Kolkata Knight Riders','Royal Challengers Bangalore','Deccan Chargers','Chennai Super Kings',
                 'Rajasthan Royals','Delhi Daredevils','Gujarat Lions','Kings XI Punjab',
                 'Sunrisers Hyderabad','Rising Pune Supergiants','Kochi Tuskers Kerala','Pune Warriors','Rising Pune Supergiant','Delhi Capitals']
                ,['MI','KKR','RCB','DC','CSK','RR','DD','GL','KXIP','SRH','RPS','KTK','PWI','RPS','DCAP'],inplace=True)

matches.replace(to_replace = 'Bangalore',value = 'Bengaluru', inplace=True)
matches.replace(to_replace = 'DD',value = 'DCAP', inplace=True)
deliveries.replace(to_replace = 'DD',value = 'DCAP', inplace=True)
def drawbar(x):
    m = pd.DataFrame(matches[x].value_counts(ascending=False))
    fig=plt.figure(figsize=(14,8))
    ax=sns.barplot(y=m.index,x=m.iloc[:,0], data=m, palette='coolwarm')
    initialx=0
    for p in ax.patches:
        ax.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()))
        initialx+=1
drawbar('city')
plt.xlabel('Number of Games', size=12)
plt.ylabel('Cities', size=12)
plt.title('Cities With Most Number of Games', size=14)
mvenue = pd.DataFrame(matches['venue'].value_counts(ascending=False))[:20]
fig=plt.figure(figsize=(14,8))
ax=sns.barplot(y=mvenue.index,x=mvenue.iloc[:,0], data=mvenue, palette='viridis')
initialx=0
plt.xlabel('Counts', size=12)
plt.ylabel('Grounds', size=12)
plt.title('Top 20 Popular Grounds', size=14)
for p in ax.patches:
    ax.text(p.get_width()*1.005,initialx+p.get_height()/4,'{:1.0f}'.format(p.get_width()))
    initialx+=1
drawbar('toss_winner')
plt.xlabel('Toss Wins', size=12)
plt.ylabel('Team Names', size=12)
plt.title('Number of Toss Wins', size=14)
fig=plt.figure(figsize=(14,8))
sns.countplot(x='season',hue='toss_decision',data=matches)
plt.title('Toss Decisions By Seasons')
drawbar('winner')
plt.xlabel('Wins', size=12)
plt.ylabel('Team Names', size=12)
plt.title('Number of Victories', size=14)
mplaofmat = pd.DataFrame(matches['player_of_match'].value_counts(ascending=False))[:20]
fig=plt.figure(figsize=(14,8))
ax=sns.barplot(y=mplaofmat.index,x=mplaofmat.iloc[:,0], data=mplaofmat, palette='viridis')
initialx=0
plt.xlabel('Counts', size=12)
plt.ylabel('Players', size=12)
plt.title('Man Of The Match Awards', size=14)
for p in ax.patches:
    ax.text(p.get_width()*1.005,initialx+p.get_height()/4,'{:1.0f}'.format(p.get_width()))
    initialx+=1
print('IPL matches have been played in',matches['city'].nunique(),'different cities and', matches['venue'].nunique(),'grounds with',matches['player_of_match'].nunique(),'unqiue players winning Man Of The Match Awards')
ump = pd.DataFrame(matches['umpire1'])
ump1 = pd.DataFrame(matches['umpire2']).rename(columns = {'umpire2':'umpire1'})
ump = ump.append(ump1, ignore_index=True)
ump = pd.DataFrame(ump['umpire1'].value_counts(ascending=False))[:20]
fig=plt.figure(figsize=(14,8))
ax=sns.barplot(y=ump.index,x=ump.iloc[:,0], data=ump, palette='coolwarm')
initialx=0
plt.xlabel('Counts', size=12)
plt.ylabel('Umpires', size=12)
plt.title('Top 20 Standing Umpires', size=14)
for p in ax.patches:
    ax.text(p.get_width()*1.005,initialx+p.get_height()/4,'{:1.0f}'.format(p.get_width()))
    initialx+=1
matches[matches['win_by_runs'] == matches['win_by_runs'].max()][['season','city','date','team1','team2','toss_winner','player_of_match','win_by_runs']]
matches[matches['win_by_wickets'] == matches['win_by_wickets'].max()]
teamgames = pd.DataFrame(matches['team1'])
teamgames1 = pd.DataFrame(matches['team2']).rename(columns = {'team2':'team1'})
teamgames = teamgames.append(teamgames1, ignore_index=True)
teamgames = pd.DataFrame(teamgames['team1'].value_counts(ascending=False)).reset_index().rename(columns= {'index':'team','team1':'matches'})
teamwins = pd.DataFrame(matches['winner'].value_counts()).reset_index().rename(columns= {'index':'team','winner':'wins'})
teamwinper = pd.merge(teamgames,
                 teamwins,
                 on='team')
teamwinper['win%'] = (teamwinper['wins']/teamwinper['matches'])*100
teamwinper = teamwinper.sort_values(by='win%',ascending=False)
teamwinper['losses'] = teamwinper['matches'] - teamwinper['wins'] 
teamwinper['win%'] = teamwinper['win%'].round(decimals=1)
fig=plt.figure(figsize=(14,8))
clrs = ['#DC460A','#F4FE4E','#0846E4','#F59E07','#3D064A','#0E6FD9','#D90ED2','#C91934','#FF0230','#E66E20','#925228','#CA3E16','#4E7FED','#53C1EE']
ax=sns.barplot(x=teamwinper['team'],y=teamwinper.iloc[:,3], data=teamwinper, palette=clrs)
initialx=0
plt.xlabel('Win Percentage', size=12)
plt.ylabel('Teams', size=12)
plt.title('Win Percentage', size=14)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=13)
fig=plt.figure(figsize=(14,8))
ax=sns.barplot(y=teamwinper['team'],x=teamwinper['matches'], data=teamwinper, color='blue')
ax1=sns.barplot(y=teamwinper['team'],x=teamwinper['wins'], data=teamwinper, color='red')
topbar = plt.Rectangle((0,0),1,1,fc="blue", edgecolor = 'none')
bottombar = plt.Rectangle((0,0),1,1,fc='red',  edgecolor = 'none')
l = plt.legend([bottombar, topbar], ['Wins', 'Mat'], loc=4, ncol = 2, prop={'size':12})
tmwin = matches[['team1','team2','toss_winner','winner']]
tmwin['tosseffect'] = np.where(tmwin['toss_winner']==tmwin['winner'], 1, 0)
tmwin['batfirstresult'] = np.where(tmwin['team1']==tmwin['winner'], 1, 0)
fig=plt.figure(figsize=(12,5))
labels = ['Toss won, match won', 
         'Toss won, match lost']
slices = [393, 363]
explode=(0.1,0)
plt.rcParams['font.size'] = 13
plt.pie(slices, labels = labels, startangle=90,shadow=True,explode=(0,0.05), autopct='%1.1f%%', colors=['#063A50','#4BC7FB'])
ig=plt.figure(figsize=(12,5))
labels = ['Match Won Batting First', 
         'Match Won Chasing']
slices = [335, 421]
plt.rcParams['font.size'] = 13
plt.pie(slices, labels = labels, startangle=90,shadow=True,explode=(0,0.05), autopct='%1.1f%%', colors=['#352087','#866FE1'])
x=['2008', '2009', '2010','2011', '2012','2013','2014','2015']
tmwintill2015 = matches[matches['season'].isin(x)]
tmwintill2015 = tmwintill2015[['team1','team2','toss_winner','winner']]
tmwintill2015['tosseffect'] = np.where(tmwintill2015['toss_winner']==tmwintill2015['winner'], 1, 0)
tmwintill2015['batfirstresult'] = np.where(tmwintill2015['team1']==tmwintill2015['winner'], 1, 0)
fig=plt.figure(figsize=(12,5))
labels = ['Match Won Batting First', 
         'Match Won Chasing']
slices = [239, 278]
plt.rcParams['font.size'] = 13
plt.title('From 2008-2015')
plt.pie(slices, labels = labels, startangle=90,shadow=True,explode=(0,0.05), autopct='%1.1f%%', colors=['#E42542','#E46C7E'])
mat = matches
mat['toss_decision1'] = np.where(mat['toss_decision']=='bat', 1, 0)
toss_teamwise = pd.DataFrame(mat.groupby('toss_winner')['toss_decision1'].sum()).reset_index()
temp = pd.DataFrame(mat['toss_winner'].value_counts()).reset_index().rename(columns={'index':'toss_winner','toss_winner':'counts'})
toss_teamwise = pd.merge(toss_teamwise, temp, on='toss_winner')
toss_teamwise = toss_teamwise.rename(columns={'toss_decision1':'bat first'})
toss_teamwise['bowl first'] = toss_teamwise['counts'] - toss_teamwise['bat first'] 
pal=['#F3CF87','#F31727']
toss_teamwise.plot(x="toss_winner",y=["bat first","bowl first"], kind="bar", color = pal, stacked=True, figsize=(14,8)).legend(['Bat','Field'])
plt.title("Toss Decisons By Teams", size=12)
plt.xlabel("Teams")
plt.ylabel("Toss Decision")
mat['toss_win_team_win'] = np.where(mat['toss_winner']==mat['winner'],1,0)
toss_win_team_win = pd.DataFrame(mat.groupby('toss_winner')['toss_win_team_win'].sum()).reset_index()
temp1 = pd.DataFrame(mat['toss_winner'].value_counts()).reset_index().rename(columns={'index':'toss_winner','toss_winner':'counts'})
toss_win_team_win = pd.merge(toss_win_team_win, temp, on='toss_winner')
toss_win_team_win['toss_win_team_lost'] = toss_win_team_win['counts'] - toss_win_team_win['toss_win_team_win']
pal=['#09420A','#41EE46']
toss_win_team_win.plot(x="toss_winner",y=["toss_win_team_win","toss_win_team_lost"], kind="bar", color=pal, stacked=True, figsize=(14,8)).legend(['Toss Won Match Won','Toss Won Match Lost'])
plt.title("Impact of Toss Decision", size=12)
plt.xlabel("Teams")
plt.ylabel("Toss Decision Effect")
mat['bat_first_team_win'] = np.where(mat['team1']==mat['winner'],1,0)
mat['bowl_first_team_win'] = np.where(mat['team2']==mat['winner'],1,0)
bat_first_team_win = pd.DataFrame(mat.groupby('team1')['bat_first_team_win'].sum()).reset_index()
bowl_first_team_win = pd.DataFrame(mat.groupby('team1')['bowl_first_team_win'].sum()).reset_index()
bat_first_team_win = pd.merge(bat_first_team_win, bowl_first_team_win, on="team1")
temp2 = pd.DataFrame(mat['team1'].value_counts()).reset_index().rename(columns={'index':'team1','team1':'bat_first'})
temp3 = pd.DataFrame(mat['team2'].value_counts()).reset_index().rename(columns={'index':'team1','team2':'bowl_first'})
bat_first_team_win = pd.merge(bat_first_team_win, temp2, on="team1")
bat_first_team_win = pd.merge(bat_first_team_win, temp3, on="team1")
bat_first_team_win['bat_first_lost'] = bat_first_team_win['bat_first'] - bat_first_team_win['bat_first_team_win']
bat_first_team_win['bowl_first_lost'] = bat_first_team_win['bowl_first'] - bat_first_team_win['bowl_first_team_win']
pal=['#2954EC','#8596D1']
bat_first_team_win.plot(x="team1",y=["bat_first_team_win","bat_first_lost"], kind="bar", color=pal, stacked=True, figsize=(14,8)).legend(['Won Batting First','Lost Batting First'])
plt.title("Winning Probability While Batting First", size=12)
plt.xlabel("Teams")
plt.ylabel("Batting First Impact")
pal = ['#5F1369','#D479E0']
bat_first_team_win.plot(x="team1",y=["bowl_first_team_win","bowl_first_lost"], kind="bar", color =pal, stacked=True, figsize=(14,8)).legend(['Won Bowling First','Lost Bowling First'])
plt.title("Winning Probability While Bowling First", size=12)
plt.xlabel("Teams")
plt.ylabel("Bowling First Impact")
def team_comp(x,y,d,e):
    a=[x,y]
    mat_new1 = mat[mat['team1'].isin(a) & mat['team2'].isin(a)][['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner',
       'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs',
       'win_by_wickets', 'player_of_match']]
    b=[d,e]
    mat_new2 = mat[mat['team1'].isin(b) & mat['team2'].isin(b)][['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner',
       'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs',
       'win_by_wickets', 'player_of_match']]
    fig, (ax1,ax2) =plt.subplots(1,2, figsize=(16,6))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    sns.countplot(x='winner', data=mat_new1, ax=ax1)
    plt.yticks(np.arange(0,20,1))
    
    sns.countplot(x='winner', data=mat_new2, ax=ax2)
    plt.yticks(np.arange(0,20,1))
team_comp('MI','CSK','RR','RCB')
def team_comp(x,y,d,e):
    a=[x,y]
    mat_new1 = mat[mat['team1'].isin(a) & mat['team2'].isin(a)][['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner',
       'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs',
       'win_by_wickets', 'player_of_match']]
    b=[d,e]
    mat_new2 = mat[mat['team1'].isin(b) & mat['team2'].isin(b)][['id', 'season', 'city', 'date', 'team1', 'team2', 'toss_winner',
       'toss_decision', 'result', 'dl_applied', 'winner', 'win_by_runs',
       'win_by_wickets', 'player_of_match']]
    fig, (ax1,ax2) =plt.subplots(1,2, figsize=(16,6))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
    sns.countplot(x='season', hue='winner', data=mat_new1, ax=ax1)
    ax1.set_yticks(np.arange(0,6,1))
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Win Count')
#     ax1.legend(bbox_to_anchor = (0.41,0.52,0.6,0.5))
    ax1.legend(loc=1)
    ax1.tick_params(labelsize=10)
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)
    
    sns.countplot(x='season', hue='winner', data=mat_new2, ax=ax2)
    ax2.set_yticks(np.arange(0,6,1))
    plt.xticks(rotation='vertical')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Win Count')
    ax2.legend(loc=1)
    ax2.tick_params(labelsize=10)
    for tick in ax2.get_xticklabels():
        tick.set_rotation(90)
team_comp('MI','CSK','RR','RCB')
deliv = deliveries
a = mat.rename(columns={'id':'match_id'})
deliv = pd.merge(deliv, a[['match_id','season']], on='match_id')
deliv['out'] = np.where(pd.isnull(deliv['player_dismissed']),0,1)
deliv.head()
overall_bat = deliv.groupby('batting_team').sum().reset_index()
overall_bat['runs/wick'] = (overall_bat['total_runs']/overall_bat['out']).round(1)
overall_bat = overall_bat.sort_values(by='runs/wick', ascending=False)

temp_overall_bat = deliv.groupby('batting_team').count().reset_index()
temp_overall_bat = temp_overall_bat.rename(columns={'ball':'ball_count'})
overall_bat = pd.merge(overall_bat, temp_overall_bat[['batting_team','ball_count']], on='batting_team')

overall_bat['team_strikerate'] = ((overall_bat['total_runs']/overall_bat['ball_count'])*100).round()
overall_bat.head()



overall_bowl = deliv.groupby('bowling_team').sum().reset_index()
overall_bowl['runs/wick'] = (overall_bowl['total_runs']/overall_bowl['out']).round(1)
overall_bowl = overall_bowl.sort_values(by='runs/wick', ascending=False)


temp_overall_bowl = deliv.groupby('bowling_team').count().reset_index()
temp_overall_bowl = temp_overall_bowl.rename(columns={'ball':'ball_count'})
overall_bowl = pd.merge(overall_bowl, temp_overall_bowl[['bowling_team','ball_count']], on='bowling_team')


overall_bowl['team_strikerate'] = ((overall_bowl['total_runs']/overall_bowl['ball_count'])*100).round()
overall_bowl.head()
fig, ([ax1,ax2],[ax3,ax4]) =plt.subplots(2,2, figsize=(14,8))
pal = ['#1F83F7']
sns.barplot(y='runs/wick',x='batting_team', data=overall_bat, palette=pal, ax=ax1)
initialx=0
ax1.set_xlabel('Teams')
ax1.set_ylabel('Batting Average')
ax1.set_yticks(np.arange(18,34,1))
ax1.set_ylim([18,34])
plt.tight_layout()
for p in ax1.patches:
    ax1.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
    
sns.barplot(y='team_strikerate',x='batting_team', data=overall_bat.sort_values(by='team_strikerate', ascending=False), palette=pal, ax=ax2)
initialx=0
ax2.set_xlabel('Teams')
ax2.set_ylabel('Batting Strike Rate')
ax2.set_yticks(np.arange(110,140,2))
ax2.set_ylim([110,140])
plt.tight_layout()
for p in ax2.patches:
    ax2.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
    
sns.barplot(y='runs/wick',x='bowling_team', data=overall_bowl.sort_values(by='runs/wick', ascending=False), palette=pal, ax=ax3)
initialx=0
ax3.set_xlabel('Teams')
ax3.set_ylabel('Bowling Average')
ax3.set_yticks(np.arange(18,36,1))
ax3.set_ylim([18,36])
plt.tight_layout()
for p in ax3.patches:
    ax3.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
    
sns.barplot(y='team_strikerate',x='bowling_team', data=overall_bowl.sort_values(by='team_strikerate', ascending=False), palette=pal, ax=ax4)
initialx=0
ax4.set_xlabel('Teams')
ax4.set_ylabel('Bowling Strike Rate')
ax4.set_yticks(np.arange(116,140,2))
ax4.set_ylim([115,148])
plt.tight_layout()
for p in ax4.patches:
    ax4.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
mat_new1 = pd.DataFrame(mat['team1'])
mat_new2 = pd.DataFrame(mat['team2']).rename(columns={'team2':'team1'})
mat_new1 = pd.concat([mat_new1, mat_new2])
mat_count = pd.DataFrame(mat_new1['team1'].value_counts()).reset_index().rename(columns={'index':'bowling_team','team1':'match_count'})
overall_bowl = pd.merge(overall_bowl, mat_count, on='bowling_team')

overall_bowl['wideruns_game'] = (overall_bowl['wide_runs']/overall_bowl['match_count']).round(1)
overall_bowl['byeruns_game'] = (overall_bowl['bye_runs']/overall_bowl['match_count']).round(2)
overall_bowl['legbyeruns_game'] = (overall_bowl['legbye_runs']/overall_bowl['match_count']).round(2)
overall_bowl['noballruns_game'] = (overall_bowl['noball_runs']/overall_bowl['match_count']).round(2)
overall_bowl['extraruns_game'] = (overall_bowl['extra_runs']/overall_bowl['match_count']).round(1)
overall_bowl['wideno_game'] = ((overall_bowl['wide_runs']+overall_bowl['noball_runs'])/(overall_bowl['match_count'])).round(1)
overall_bowl
fig, ([ax1,ax2],[ax3,ax4]) =plt.subplots(2,2, figsize=(14,8))
sns.barplot(y='wideruns_game',x='bowling_team', data=overall_bowl.sort_values(by='wideruns_game', ascending=False), palette='viridis', ax=ax1)
initialx=0
ax1.set_xlabel('Teams')
ax1.set_ylabel('Wide Runs/Game')
plt.tight_layout()
for p in ax1.patches:
    ax1.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
    
sns.barplot(y='noballruns_game',x='bowling_team', data=overall_bowl.sort_values(by='noballruns_game', ascending=False), palette='viridis', ax=ax2)
initialx=0
ax2.set_xlabel('Teams')
ax2.set_ylabel('No Ball Runs/Game')
plt.tight_layout()
for p in ax2.patches:
    ax2.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
    
sns.barplot(y='wideno_game',x='bowling_team', data=overall_bowl.sort_values(by='wideno_game', ascending=False), palette='viridis', ax=ax3)
initialx=0
ax3.set_xlabel('Teams')
ax3.set_ylabel('Wide + No Ball Extras')
plt.tight_layout()
for p in ax3.patches:
    ax3.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
    
sns.barplot(y='extraruns_game',x='bowling_team', data=overall_bowl.sort_values(by='extraruns_game', ascending=False), palette='viridis', ax=ax4)
initialx=0
ax4.set_xlabel('Teams')
ax4.set_ylabel('Total Extra Runs/Game')
plt.tight_layout()
for p in ax4.patches:
    ax4.annotate(str(p.get_height()), (p.get_x() * 1.015, p.get_height() * 1.005), size=10)
overall_seasonbat = deliv.groupby('season').sum().reset_index()
overall_seasonbat['runs/wick'] = (overall_seasonbat['total_runs']/overall_seasonbat['out']).round(1)

temp_overall_season = deliv.groupby('season').count().reset_index()
temp_overall_season = temp_overall_season.rename(columns={'ball':'ball_count'})
overall_seasonbat = pd.merge(overall_seasonbat, temp_overall_season[['season','ball_count']], on='season')
overall_seasonbat['team_strikerate'] = ((overall_seasonbat['total_runs']/overall_seasonbat['ball_count'])*100).round()

fig, (ax1,ax2) =plt.subplots(1,2, figsize=(18,8))
sns.lineplot(y='runs/wick',x='season', data=overall_seasonbat, color='r', ax=ax1)
plt.subplots_adjust(left=None, bottom=None, right=1, top=None, wspace=None, hspace=None)
initialx=0
fig.suptitle('Season Wise Batting Average and Batting StrikeRate')
ax1.set_xlabel('Season')
ax1.set_ylabel('Batting Average')
# plt.tight_layout()
ax1.set_yticks(np.arange(20,32,1))
ax1.set_xticks(np.arange(2008,2020,1))
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
    
sns.lineplot(y='team_strikerate',x='season', data=overall_seasonbat, palette='viridis', ax=ax2)
initialx=0
ax2.set_xlabel('Season')
ax2.set_ylabel('Batting StrikeRate')
ax2.set_yticks(np.arange(120,146,1))
ax2.set_xticks(np.arange(2008,2020,1))
# plt.tight_layout()
for tick in ax2.get_xticklabels():
    tick.set_rotation(45)
overall_batseason = deliv.groupby(['season','batting_team']).sum().reset_index()
overall_batseason['runs/wick'] = (overall_batseason['total_runs']/overall_batseason['out']).round(1)

temp_overall_batseason = deliv.groupby(['season','batting_team']).count().reset_index()
temp_overall_batseason = temp_overall_batseason.rename(columns={'ball':'ball_count'})
overall_batseason = pd.merge(overall_batseason, temp_overall_batseason[['batting_team','ball_count']], on='batting_team')
overall_batseason['team_strikerate'] = ((overall_batseason['total_runs']/overall_batseason['ball_count'])*100).round()



fig, (ax1,ax2) =plt.subplots(2,1, figsize=(18,15))
pal=['#E8F71F','#7F0EF5','#6F0011','#260635','#839192','#0B208D','#F1220D','#09BBF2','#48C9B0','#1C2833','#FF8300','#28FF00','#AA7F98',]
sns.lineplot(x='season', y='runs/wick', hue='batting_team', palette=pal, data=overall_batseason, ax=ax1)
ax1.legend(title = 'Teams', loc = 1, fontsize = 15.5)
ax1.set_xlim([2008,2022])
ax1.set_xlabel('Season')
ax1.set_ylabel('Batting Average')

sns.lineplot(x='season', y='team_strikerate', hue='batting_team', palette=pal, ci=None, data=overall_batseason, ax=ax2)
ax2.legend(title = 'Teams', loc = 1, fontsize = 15.5)
ax2.set_xlim([2008,2022])
ax2.set_xlabel('Season')
ax2.set_ylabel('Batting StrikeRate')
batsman = deliv.groupby('batsman').sum().reset_index()
temp10 = deliv.groupby('batsman').count().reset_index().rename(columns={'ball':'ball_count'})
batsman = pd.merge(batsman, temp10[['batsman','ball_count']], on='batsman')
# batsman['batting_SR'] = ((batsman['batsman_runs']/batsman['ball_count'])*100).round(2)
temp11 = pd.DataFrame(deliv['player_dismissed'].value_counts()).reset_index().rename(columns={'index':'batsman','player_dismissed':'counts'})
batsman = pd.merge(batsman, temp11, on='batsman').rename(columns={'counts':'bat_out'})
batsman['batting_average'] = (batsman['batsman_runs']/batsman['bat_out']).round(2)
tempalls = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==4).sum()).reset_index().rename(columns={'batsman_runs':'4s'})
temp6s = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==6).sum()).reset_index().rename(columns={'batsman_runs':'6s'})
temp1s = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==1).sum()).reset_index().rename(columns={'batsman_runs':'1s'})
temp2s = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==2).sum()).reset_index().rename(columns={'batsman_runs':'2s'})
temp3s = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==3).sum()).reset_index().rename(columns={'batsman_runs':'3s'})
temp5s = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==5).sum()).reset_index().rename(columns={'batsman_runs':'5s'})
temp7s = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==7).sum()).reset_index().rename(columns={'batsman_runs':'7s'})
temp0s = deliv.groupby('batsman')['batsman_runs'].agg(lambda x: (x==0).sum()).reset_index().rename(columns={'batsman_runs':'0s'})
tempalls = pd.merge(tempalls, temp6s, on='batsman')
tempalls = pd.merge(tempalls, temp1s, on='batsman')
tempalls = pd.merge(tempalls, temp2s, on='batsman')
tempalls = pd.merge(tempalls, temp3s, on='batsman')
tempalls = pd.merge(tempalls, temp5s, on='batsman')
tempalls = pd.merge(tempalls, temp7s, on='batsman')
tempalls = pd.merge(tempalls, temp0s, on='batsman')

batsman = pd.merge(batsman, tempalls, on='batsman')
batsman = batsman[['batsman','wide_runs', 'bye_runs', 'legbye_runs', 'noball_runs', 'penalty_runs',
       'batsman_runs', 'extra_runs', 'total_runs', 'ball_count','bat_out', 'batting_average', '4s', '6s',
       '1s', '2s', '3s', '5s', '7s', '0s']]
batsman['actual_ballsfaced'] = batsman['ball_count'] - batsman['wide_runs']
batsman['dots'] = batsman['actual_ballsfaced'] - (batsman['4s'] + batsman['6s'] + batsman['1s'] + batsman['2s'] + batsman['3s']
                                           + batsman['5s'] + batsman['7s']
                                           + batsman['legbye_runs'] + batsman['bye_runs'])
batsman['batting_SR'] = ((batsman['batsman_runs']/(batsman['ball_count'] - batsman['wide_runs']))*100).round(2)
batsman['dotper'] = ((batsman['dots']/batsman['actual_ballsfaced'])*100).round(2)
batsman['4sper'] = ((batsman['4s']/batsman['actual_ballsfaced'])*100).round(2)
batsman['6sper'] = ((batsman['6s']/batsman['actual_ballsfaced'])*100).round(2)
batsman.head()
fig, (ax1,ax2) =plt.subplots(1,2, figsize=(14,7))
sns.regplot(x='batting_SR', y='dotper', data=batsman, color='red', ax=ax1)
plt.subplots_adjust(left=None, bottom=None, right=1, top=None, wspace=None, hspace=1)
ax1.legend(title = 'Dot Ball % vs Batting SR', loc = 1, fontsize = 15.5)
ax1.set_xlabel('Batting SR')
ax1.set_ylabel('Dot Ball Percentage')

sns.regplot(x='batting_SR', y='batting_average', data=batsman, color='orange', ax=ax2)
ax2.legend(title = 'Batting Average vs Batting SR', loc = 0, fontsize = 15.5)
ax2.set_xlabel('Batting SR')
ax2.set_ylabel('Batting Average')
fig, ([ax1,ax2],[ax3,ax4], [ax5,ax6], [ax7,ax8]) =plt.subplots(4,2, figsize=(14,18))
sns.barplot(y='batsman',x='batsman_runs', data=batsman
            .sort_values(by='batsman_runs', ascending=False)[:10], palette='viridis', ax=ax1)
initialx=0
ax1.set_xlabel('Runs')
ax1.set_ylabel('Batsman')
ax1.tick_params(axis="y", labelsize=10)
ax1.set_xlim([4000,5600])
ax1.set_title('Top Run Scorers', size=12)
plt.tight_layout()
for p in ax1.patches:
    ax1.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1
    

sns.barplot(y='batsman',x='batting_average', data=batsman[batsman['batsman_runs']>500]
            .sort_values(by='batting_average', ascending=False)[:10], palette='viridis', ax=ax2)
initialx=0
ax2.set_xlabel('Batting Average')
ax2.set_ylabel('Batsman')
ax2.tick_params(axis="y", labelsize=10)
ax2.set_xlim([30,50])
ax2.set_title('Batting Average (Min 500 Runs)', size=12)
plt.tight_layout()
for p in ax2.patches:
    ax2.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1

    
    
sns.barplot(y='batsman',x='batting_SR', data=batsman[batsman['batsman_runs']>500]
            .sort_values(by='batting_SR', ascending=False)[:10], palette='viridis', ax=ax3)
initialx=0
ax3.set_xlabel('Batting Strike Rate')
ax3.set_ylabel('Batsman')
ax3.tick_params(axis="y", labelsize=10)
ax3.set_xlim([145,200])
ax3.set_title('Batting SR (Min 500 Runs)', size=12)
plt.tight_layout()
for p in ax3.patches:
    ax3.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1
    
sns.barplot(y='batsman',x='dotper', data=batsman[batsman['batsman_runs']>500]
            .sort_values(by='dotper', ascending=False)[:10], palette='viridis', ax=ax4)
initialx=0
ax4.set_xlabel('Dot Ball %')
ax4.set_ylabel('Batsman')
ax4.tick_params(axis="y", labelsize=10)
ax4.set_xlim([35,50])
ax4.set_title('Dot Ball % (Min 500 Runs)', size=12)
plt.tight_layout()
for p in ax4.patches:
    ax4.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1
    
sns.barplot(y='batsman',x='4s', data=batsman
            .sort_values(by='4s', ascending=False)[:10], palette='viridis', ax=ax5)
initialx=0
ax5.set_xlabel('Fours')
ax5.set_ylabel('Batsman')
ax5.tick_params(axis="y", labelsize=10)
ax5.set_xlim([350,550])
ax5.set_title('Number of Fours', size=12)
plt.tight_layout()
for p in ax5.patches:
    ax5.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1
    
sns.barplot(y='batsman',x='6s', data=batsman
            .sort_values(by='6s', ascending=False)[:10], palette='viridis', ax=ax6)
initialx=0
ax6.set_xlabel('Sixes')
ax6.set_ylabel('Batsman')
ax6.tick_params(axis="y", labelsize=10)
ax6.set_xlim([150,340])
ax6.set_title('Number of Sixes', size=12)
plt.tight_layout()
for p in ax6.patches:
    ax6.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1

sns.barplot(y='batsman',x='4sper', data=batsman[batsman['batsman_runs']>500]
            .sort_values(by='4sper', ascending=False)[:10], palette='viridis', ax=ax7)
initialx=0
ax7.set_xlabel('4s %')
ax7.set_ylabel('Batsman')
ax7.tick_params(axis="y", labelsize=10)
ax7.set_xlim([12,25])
ax7.set_title('Fours Percentage (Min 500 Runs, Based on No. of Balls Faced)', size=12)
plt.tight_layout()
for p in ax7.patches:
    ax7.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1

sns.barplot(y='batsman',x='6sper', data=batsman[batsman['batsman_runs']>500]
            .sort_values(by='6sper', ascending=False)[:10], palette='viridis', ax=ax8)
initialx=0
ax8.set_xlabel('6s %')
ax8.set_ylabel('Batsman')
ax8.tick_params(axis="y", labelsize=10)
ax8.set_xlim([6,20])
ax8.set_title('Sixes Percentage (Min 500 Runs, Based on No. of Balls Faced)', size=12)
plt.tight_layout()
for p in ax8.patches:
    ax8.text(p.get_width()*1.005,initialx+p.get_height()/5,'{:1.0f}'.format(p.get_width()),size=10)
    initialx+=1
