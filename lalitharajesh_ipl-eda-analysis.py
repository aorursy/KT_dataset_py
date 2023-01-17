from IPython.display import Image
Image(filename='../input/ipl-logo/IPL.png')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("../input/ipl/matches.csv")
len(data)
df = pd.read_csv("../input/ipl/deliveries.csv")
len(df)
#list of column names
data.columns
#length of the Columns
len(data.columns)
data.head()
#total number of teams participating (Team names)
team1_names = data.team1.unique()
team1_names.tolist()
team2_names = data.team2.unique()
team2_names.tolist()
total_teams = list(set(team1_names) & set(team2_names))
print (total_teams)
#Total number of teams participating (Team size)
print (len(total_teams))
#Details on Toss won by each team, Total Matches played so far, total matches being won list.
team_stats = pd.DataFrame({'Total Matches played': data.team1.value_counts() + data.team2.value_counts(), 'Total won': data.winner.value_counts(), 'Toss won': data.toss_winner.value_counts(), 
                          'Total lost': ((data.team1.value_counts() + data.team2.value_counts()) - data.winner.value_counts())})
team_stats = team_stats.reset_index()
team_stats.rename(columns = {'index':'Teams'}, inplace = True)
team_stats
#Maximum Toss Won:
plt.subplots(figsize=(10,7))
ax=data['toss_winner'].value_counts().plot.barh(width=0.8)
plt.title("Maximum Toss Won")
#Teams who had won Toss and Won the match
Tosswin_matchwin=data[data['toss_winner']==data['winner']]
slices=[len(Tosswin_matchwin),(636-len(Tosswin_matchwin))]
labels=['Yes','No']
plt.pie(slices,labels=labels,startangle=90,shadow=True,explode=(0,0),autopct='%1.1f%%',colors=['g','r'])
plt.title("Teams who had won Toss and Won the match")
fig = plt.gcf()
fig.set_size_inches(5,5)
plt.show()
#Plotting the above data Analysis - Total Won
Total_won = data.winner.value_counts()
ax = Total_won.plot(kind='bar', title ="No. of wins by each team - Overall",figsize=(10,7),legend=True, fontsize=12)
ax.set_xlabel("Teams",fontsize=12)
ax.set_ylabel("count",fontsize=12)
winloss = team_stats['Total won'] / team_stats['Total Matches played']
winloss = pd.DataFrame({'Winloss Ratio': team_stats['Total won'] / team_stats['Total Matches played']})
winloss= winloss.round(2)
team_stats = team_stats.join(winloss)
team_stats
winloss_types = ['Tie/Wins','Tie/Loss','NR']
played =0
winloss_list1 = []

for t in data.team1.unique():
    for wl in winloss_types:
        Tie = 0
        if wl == 'Tie/Wins':             
            Tie = len(data[((data.result == 'tie') & (data.winner == t)) & ((data.team1 == t) | (data.team2 == t))])            
        elif wl == 'Tie/Loss':
            Tie = len(data[((data.result == 'tie') & (data.winner != t)) & ((data.team1 == t) | (data.team2 == t))])
        elif wl == 'NR':
            Tie = len(data[((data.result == 'no result') & (data.winner != t)) & ((data.team1 == t) | (data.team2 == t))])
        if (Tie!= 0):
            winloss_list1.append(Tie)
        else:
            winloss_list1.append(0)
            
winloss_ser1 = pd.Series(winloss_list1, index=pd.MultiIndex.from_product([data.team1.unique(), winloss_types]))
winloss_ser1.unstack()
winloss_types = ['WinlossTosswin','WinlossBat','WinlossField','winlossTie']
played =0
winloss_list = []
for t in data.team1.unique():
    for wl in winloss_types:
        won, loss, Tie = 0, 0, 0
        if wl == 'WinlossTosswin':
            won = len(data[(data.toss_winner == t) & (data.winner == t)])
            played = len(data[((data.team1 == t) | (data.team2 == t)) &
                 (((data.toss_winner == t) & (data.winner == t))
                  | ((data.toss_winner != t) & (data.winner != t)))])
        if wl == 'WinlossBat':
            won = len(data[(data.winner == t) & (((data.toss_winner == t) & (data.toss_decision == 'bat'))
                  | ((data.toss_winner != t) & (data.toss_decision == 'field')))])
            played = len(data[((data.team1 == t) | (data.team2 == t)) &
                 (((data.toss_winner == t) & (data.toss_decision == 'bat'))
                  | ((data.toss_winner != t) & (data.toss_decision == 'field')))])
        elif wl == 'WinlossField':
            won = len(data[(data.winner == t) & (((data.toss_winner == t) & (data.toss_decision == 'field'))
                  | ((data.toss_winner != t) & (data.toss_decision == 'bat')))])
            played = len(data[((data.team1 == t) | (data.team2 == t)) &
                 (((data.toss_winner == t) & (data.toss_decision == 'field'))
                  | ((data.toss_winner != t) & (data.toss_decision == 'bat')))])
        elif wl == 'winlossTie':
                won =len(data[((data.result == 'tie') & (data.winner == t)) & ((data.team1 == t) | (data.team2 == t))])
                played = len(data[((data.team1 == t) | (data.team2 == t))])
        
        if((won != 0) & (played != 0)):
            winloss_list.append(np.round(won/played,2))
        else:
            winloss_list.append(0)
        
winloss_ser2 = pd.Series(winloss_list, index=pd.MultiIndex.from_product([data.team1.unique(), winloss_types]))
winloss_ser2.unstack()
Teams=['Chennai Super Kings','Deccan Chargers','Delhi Daredevils','Gujarat Lions','Kings XI Punjab','Kochi Tuskers Kerala','Kolkata Knight Riders','Mumbai Indians','Pune Warriors','Rajasthan Royals','Rising Pune Supergiants','Royal Challengers Bangalore','Sunrisers Hyderabad']
team_stats1 = pd.DataFrame({
         'WinlossTosswin':[0.60,0.46,0.41,0.55,0.44,0.50,0.54,0.60,0.15,0.57,0.38,0.45,0.54],
         'WinlossBat': [0.58,0.42,0.32,0.17, 0.41,0.29,0.44,0.57,0.30,0.46,0.00,0.44,0.49],
         'WinlossField':[0.63,0.34,0.50,0.80,0.54,0.57,0.59,0.58,0.23,0.59,0.71,0.56,0.64],
         'WinlossTie':[0.00,0.00,0.00,0.00,0.01,0.00,0.00,0.00,0.00,0.02,0.00,0.01,0.02],
         'Tie/Loss':[1,0,1,0,0,0,2,0,0,1,0,1,0],
         'Tie/Win': [0,0,0,0,2,0,0,0,0,2,0,1,1],
         'NR': [0,0,2,0,0,0,0,0,1,1,0,2,0]},index=Teams)
team_stats1 = team_stats1.reset_index()
team_stats1 = team_stats1.rename(index=str, columns={"index": "Teams"})
team_stats1

Total_updates = pd.merge(team_stats,team_stats1, on='Teams')
Total_updates
#Calculating the Winloss for playing in home ground. (WINLOSS-HOME)
home_winner_CSK = len(data[((data.team1 == 'Chennai Super Kings') | (data.team2 == 'Chennai Super Kings')) & ((data.city == 'Chennai') & (data.winner == 'Chennai Super Kings'))])
home_winner_KKR = len(data[((data.team1 == 'Kolkata Knight Riders') | (data.team2 == 'Kolkata Knight Riders')) & ((data.city == 'Kolkata') & (data.winner == 'Kolkata Knight Riders'))])
home_winner_RR = len(data[((data.team1 == 'Rajasthan Royals') | (data.team2 == 'Rajasthan Royals')) & ((data.city == 'Jaipur') & (data.winner == 'Rajasthan Royals'))])
home_winner_MI = len(data[((data.team1 == 'Mumbai Indians') | (data.team2 == 'Mumbai Indians')) & ((data.city == 'Mumbai') & (data.winner == 'Mumbai Indians'))])
home_winner_DC = len(data[((data.team1 == 'Deccan Chargers') | (data.team2 == 'Deccan Chargers')) & ((data.city == 'Hyderabad') & (data.winner == 'Deccan Chargers'))])
home_winner_RCB = len(data[((data.team1 == 'Royal Challengers Bangalore') | (data.team2 == 'Royal Challengers Bangalore')) & ((data.city == 'Bangalore') & (data.winner == 'Royal Challengers Bangalore'))])
home_winner_DD = len(data[((data.team1 == 'Delhi Daredevils') | (data.team2 == 'Delhi Daredevils')) & ((data.city == 'Delhi') & (data.winner == 'Delhi Daredevils'))])
home_winner_KT = len(data[((data.team1 == 'Kochi Tuskers Kerala') | (data.team2 == 'Kochi Tuskers Kerala')) & ((data.city == 'Kochi') & (data.winner == 'Kochi Tuskers Kerala'))])
home_winner_PW = len(data[((data.team1 == 'Pune Warriors') | (data.team2 == 'Pune Warriors')) & ((data.city == 'Pune') & (data.winner == 'Pune Warriors'))])
home_winner_SH = len(data[((data.team1 == 'Sunrisers Hyderabad') | (data.team2 == 'Sunrisers Hyderabad')) & ((data.city == 'Hyderabad') & (data.winner == 'Sunrisers Hyderabad'))])
home_winner_GL = len(data[((data.team1 == 'Gujarat Lions') | (data.team2 == 'Gujarat Lions')) & ((data.city == 'Rajkot') & (data.winner == 'Gujarat Lions'))])
home_winner_RPS = len(data[((data.team1 == 'Rising Pune Supergiants') | (data.team2 == 'Rising Pune Supergiants')) & ((data.city == 'Pune') & (data.winner == 'Rising Pune Supergiants'))])
home_winner_KXP = len(data[((data.team1 == 'Kings XI Punjab') | (data.team2 == 'Kings XI Punjab')) & ((data.city == 'Chandigarh') & (data.winner == 'Kings XI Punjab'))])


total_matches_played_CSK = len(data[((data.team1 == 'Chennai Super Kings') | (data.team2 == 'Chennai Super Kings')) & (data.city == 'Chennai')])
total_matches_played_KKR = len(data[((data.team1 == 'Kolkata Knight Riders') | (data.team2 == 'Kolkata Knight Riders')) & (data.city == 'Kolkata')])
total_matches_played_RR = len(data[((data.team1 == 'Rajasthan Royals') | (data.team2 == 'Rajasthan Royals')) & (data.city == 'Jaipur')])
total_matches_played_MI = len(data[((data.team1 == 'Mumbai Indians') | (data.team2 == 'Mumbai Indians')) & (data.city == 'Mumbai')])
total_matches_played_DC = len(data[((data.team1 == 'Deccan Chargers') | (data.team2 == 'Deccan Chargers')) & (data.city == 'Hyderabad')])
total_matches_played_RCB = len(data[((data.team1 == 'Royal Challengers Bangalore') | (data.team2 == 'Royal Challengers Bangalore')) & (data.city == 'Bangalore')])
total_matches_played_DD = len(data[((data.team1 == 'Delhi Daredevils') | (data.team2 == 'Delhi Daredevils')) & (data.city == 'Delhi')])
total_matches_played_KT = len(data[((data.team1 == 'Kochi Tuskers Kerala') | (data.team2 == 'Kochi Tuskers Kerala')) & (data.city == 'Kochi')])
total_matches_played_PW = len(data[((data.team1 == 'Pune Warriors') | (data.team2 == 'Pune Warriors')) & (data.city == 'Pune')])
total_matches_played_SH = len(data[((data.team1 == 'Sunrisers Hyderabad') | (data.team2 == 'Sunrisers Hyderabad')) & (data.city == 'Hyderabad')])
total_matches_played_GL = len(data[((data.team1 == 'Gujarat Lions') | (data.team2 == 'Gujarat Lions')) & (data.city == 'Rajkot')])
total_matches_played_RPS = len(data[((data.team1 == 'Rising Pune Supergiants') | (data.team2 == 'Rising Pune Supergiants')) & (data.city == 'Pune')])
total_matches_played_KXP = len(data[((data.team1 == 'Kings XI Punjab') | (data.team2 == 'Kings XI Punjab')) & (data.city == 'Chandigarh')])

winloss_home_CSK = home_winner_CSK / total_matches_played_CSK
winloss_home_KKR = home_winner_KKR / total_matches_played_KKR
winloss_home_RR = home_winner_RR / total_matches_played_RR
winloss_home_MI = home_winner_MI / total_matches_played_MI
winloss_home_DC = home_winner_DC / total_matches_played_DC
winloss_home_RCB = home_winner_RCB / total_matches_played_RCB
winloss_home_DD = home_winner_DD / total_matches_played_DD
winloss_home_KT = home_winner_KT / total_matches_played_KT
winloss_home_PW = home_winner_PW / total_matches_played_PW
winloss_home_SH = home_winner_SH / total_matches_played_SH
winloss_home_GL = home_winner_GL / total_matches_played_GL
winloss_home_RPS = home_winner_RPS / total_matches_played_RPS
winloss_home_KXP = home_winner_KXP / total_matches_played_KXP

winlossHome=pd.Series([winloss_home_CSK,winloss_home_DC,winloss_home_DD,winloss_home_GL,winloss_home_KXP,winloss_home_KT,winloss_home_KKR,winloss_home_MI,winloss_home_PW,winloss_home_RR,winloss_home_RPS,winloss_home_RCB,winloss_home_SH],index=['Chennai Super Kings','Deccan Chargers','Delhi Daredevils','Gujarat Lions','Kings XI Punjab','Kochi Tuskers Kerala','Kolkata Knight Riders','Mumbai Indians','Pune Warriors','Rajasthan Royals','Rising Pune Supergiants','Royal Challengers Bangalore','Sunrisers Hyderabad'])
winlossHome=pd.DataFrame({'WinlossHome':winlossHome})
winlossHome=winlossHome.round(2)
winlossHome=winlossHome.reset_index()
winlossHome=winlossHome.rename(index=str, columns={"index": "Teams"})

Total_updates = pd.merge(Total_updates,winlossHome, on='Teams')
Total_updates
#Calculating the Winloss for playing in other ground. (WINLOSS-AWAY)
away_winner_CSK = len(data[((data.team1 == 'Chennai Super Kings') | (data.team2 == 'Chennai Super Kings')) & ((data.city != 'Chennai') & (data.winner == 'Chennai Super Kings'))])
away_winner_KKR = len(data[((data.team1 == 'Kolkata Knight Riders') | (data.team2 == 'Kolkata Knight Riders')) & ((data.city != 'Kolkata') & (data.winner == 'Kolkata Knight Riders'))])
away_winner_RR = len(data[((data.team1 == 'Rajasthan Royals') | (data.team2 == 'Rajasthan Royals')) & ((data.city != 'Jaipur') & (data.winner == 'Rajasthan Royals'))])
away_winner_MI = len(data[((data.team1 == 'Mumbai Indians') | (data.team2 == 'Mumbai Indians')) & ((data.city != 'Mumbai') & (data.winner == 'Mumbai Indians'))])
away_winner_DC = len(data[((data.team1 == 'Deccan Chargers') | (data.team2 == 'Deccan Chargers')) & ((data.city != 'Hyderabad') & (data.winner == 'Deccan Chargers'))])
away_winner_RCB = len(data[((data.team1 == 'Royal Challengers Bangalore') | (data.team2 == 'Royal Challengers Bangalore')) & ((data.city != 'Bangalore') & (data.winner == 'Royal Challengers Bangalore'))])
away_winner_DD = len(data[((data.team1 == 'Delhi Daredevils') | (data.team2 == 'Delhi Daredevils')) & ((data.city != 'Delhi') & (data.winner == 'Delhi Daredevils'))])
away_winner_KT = len(data[((data.team1 == 'Kochi Tuskers Kerala') | (data.team2 == 'Kochi Tuskers Kerala')) & ((data.city != 'Kochi') & (data.winner == 'Kochi Tuskers Kerala'))])
away_winner_PW = len(data[((data.team1 == 'Pune Warriors') | (data.team2 == 'Pune Warriors')) & ((data.city == 'Pune') & (data.winner != 'Pune Warriors'))])
away_winner_SH = len(data[((data.team1 == 'Sunrisers Hyderabad') | (data.team2 == 'Sunrisers Hyderabad')) & ((data.city != 'Hyderabad') & (data.winner == 'Sunrisers Hyderabad'))])
away_winner_GL = len(data[((data.team1 == 'Gujarat Lions') | (data.team2 == 'Gujarat Lions')) & ((data.city != 'Rajkot') & (data.winner == 'Gujarat Lions'))])
away_winner_RPS = len(data[((data.team1 == 'Rising Pune Supergiants') | (data.team2 == 'Rising Pune Supergiants')) & ((data.city != 'Pune') & (data.winner == 'Rising Pune Supergiants'))])
away_winner_KXP = len(data[((data.team1 == 'Kings XI Punjab') | (data.team2 == 'Kings XI Punjab')) & ((data.city != 'Chandigarh') & (data.winner == 'Kings XI Punjab'))])

total_matches_played_CSK1 = len(data[((data.team1 == 'Chennai Super Kings') | (data.team2 == 'Chennai Super Kings')) & (data.city != 'Chennai')])
total_matches_played_KKR1 = len(data[((data.team1 == 'Kolkata Knight Riders') | (data.team2 == 'Kolkata Knight Riders')) & (data.city != 'Kolkata')])
total_matches_played_RR1 = len(data[((data.team1 == 'Rajasthan Royals') | (data.team2 == 'Rajasthan Royals')) & (data.city != 'Jaipur')])
total_matches_played_MI1 = len(data[((data.team1 == 'Mumbai Indians') | (data.team2 == 'Mumbai Indians')) & (data.city != 'Mumbai')])
total_matches_played_DC1 = len(data[((data.team1 == 'Deccan Chargers') | (data.team2 == 'Deccan Chargers')) & (data.city != 'Hyderabad')])
total_matches_played_RCB1 = len(data[((data.team1 == 'Royal Challengers Bangalore') | (data.team2 == 'Royal Challengers Bangalore')) & (data.city != 'Bangalore')])
total_matches_played_DD1 = len(data[((data.team1 == 'Delhi Daredevils') | (data.team2 == 'Delhi Daredevils')) & (data.city != 'Delhi')])
total_matches_played_KT1 = len(data[((data.team1 == 'Kochi Tuskers Kerala') | (data.team2 == 'Kochi Tuskers Kerala')) & (data.city != 'Kochi')])
total_matches_played_PW1 = len(data[((data.team1 == 'Pune Warriors') | (data.team2 == 'Pune Warriors')) & (data.city != 'Pune')])
total_matches_played_SH1 = len(data[((data.team1 == 'Sunrisers Hyderabad') | (data.team2 == 'Sunrisers Hyderabad')) & (data.city != 'Hyderabad')])
total_matches_played_GL1 = len(data[((data.team1 == 'Gujarat Lions') | (data.team2 == 'Gujarat Lions')) & (data.city != 'Rajkot')])
total_matches_played_RPS1 = len(data[((data.team1 == 'Rising Pune Supergiants') | (data.team2 == 'Rising Pune Supergiants')) & (data.city != 'Pune')])
total_matches_played_KXP1 = len(data[((data.team1 == 'Kings XI Punjab') | (data.team2 == 'Kings XI Punjab')) & (data.city != 'Chandigarh')])

winloss_away_CSK = away_winner_CSK / total_matches_played_CSK1
winloss_away_KKR = away_winner_KKR / total_matches_played_KKR1
winloss_away_RR = away_winner_RR / total_matches_played_RR1
winloss_away_MI = away_winner_MI / total_matches_played_MI1
winloss_away_DC = away_winner_DC / total_matches_played_DC1
winloss_away_RCB = away_winner_RCB / total_matches_played_RCB1
winloss_away_DD = away_winner_DD / total_matches_played_DD1
winloss_away_KT = away_winner_KT / total_matches_played_KT1
winloss_away_PW = away_winner_PW / total_matches_played_PW1
winloss_away_SH = away_winner_SH / total_matches_played_SH1
winloss_away_GL = away_winner_GL / total_matches_played_GL1
winloss_away_RPS = away_winner_RPS / total_matches_played_RPS1
winloss_away_KXP = away_winner_KXP / total_matches_played_KXP1

winlossAway=pd.Series([winloss_away_CSK,winloss_away_DC,winloss_away_DD,winloss_away_GL,winloss_away_KXP,winloss_away_KT,winloss_away_KKR,winloss_away_MI,winloss_away_PW,winloss_away_RR,winloss_away_RPS,winloss_away_RCB,winloss_away_SH],index=['Chennai Super Kings','Deccan Chargers','Delhi Daredevils','Gujarat Lions','Kings XI Punjab','Kochi Tuskers Kerala','Kolkata Knight Riders','Mumbai Indians','Pune Warriors','Rajasthan Royals','Rising Pune Supergiants','Royal Challengers Bangalore','Sunrisers Hyderabad'])
winlossAway= pd.DataFrame({'WinlossAway':winlossAway})
winlossAway=winlossAway.round(2)
winlossAway= winlossAway.reset_index()
winlossAway= winlossAway.rename(index=str, columns={"index": "Teams"})
Record_Summary = pd.merge(Total_updates,winlossAway, on='Teams')
Record_Summary
df.head()
df.columns
df.dtypes
df.batsman.unique()
batsmen_summary = df.groupby("batsman").agg({'ball': 'count','batsman_runs': 'sum'})
batsmen_summary.rename(columns={'ball':'balls', 'batsman_runs': 'runs'}, inplace=True)
batsmen_summary = batsmen_summary.sort_values(['balls','runs'], ascending=False)[:20]
batsmen_summary.head(10)
batsmen_summary['batting_strike_rate'] = batsmen_summary['runs']/batsmen_summary['balls'] * 100
batsmen_summary['batting_strike_rate'] = batsmen_summary['batting_strike_rate'].round(2)
batsmen_summary.head(10)
df.dismissal_kind.unique()
print("Top dismissal_kind types")
index = df['dismissal_kind'].value_counts().index.tolist()
count = df['dismissal_kind'].value_counts().tolist()
pd.DataFrame({'dismissal_kind': index[1:], 'total_num':count[1:]})
#utility function used later
def trybuild(lookuplist, buildlist):
    alist = []
    for i in buildlist.index:
        try:
            #print(i)
            alist.append(lookuplist[i])
            #print(alist)
        except KeyError:
            #print('except')
            alist.append(0)
    return alist
alist = []
for dk in df.dismissal_kind.unique():
    for dk in ['nan','hit wicket', 'retired hurt', 'obstructing the field']:
        lookuplist = df[df.dismissal_kind == dk].groupby('player_dismissed')['player_dismissed'].count()
        batsmen_summary[dk] = trybuild(lookuplist, batsmen_summary)
        try:
            alist.append(lookuplist[dk])
        except KeyError:
            alist.append(0)
TopBatsman = batsmen_summary.sort_values(['balls','runs'], ascending=False)[:20]
TopBatsman
alist = []
for r in df.batsman_runs.unique():
    lookuplist = df[df.batsman_runs == r].groupby('batsman')['batsman'].count()
    batsmen_summary[str(r) + 's'] = trybuild(lookuplist, batsmen_summary)
    try:
        alist.append(lookuplist[dk])
    except KeyError:
        alist.append(0)
TopBatsman = batsmen_summary.sort_values(['balls','runs'], ascending=False)[:20]
TopBatsman.head()
#Build a dictionary of Matches player by each batsman
played = {}
def BuildPlayedDict(x):
    #print(x.shape, x.shape[0], x.shape[1])
    for p in x.batsman.unique():
        if p in played:
            played[p] += 1
        else:
            played[p] = 1

df.groupby('match_id').apply(BuildPlayedDict)
import operator
TopBatsman['matches_played'] = [played[p] for p in TopBatsman.index]
TopBatsman['average']= TopBatsman['runs']/TopBatsman['matches_played']

TopBatsman['6s/match'] = TopBatsman['6s']/TopBatsman['matches_played']  
TopBatsman['6s/match'].median()

TopBatsman['4s/match'] = TopBatsman['4s']/TopBatsman['matches_played']  
TopBatsman['4s/match']
TopBatsman
TopBatsman1 = pd.DataFrame({'balls': TopBatsman.balls, 'runs': TopBatsman.runs,'Total Matches played':TopBatsman['matches_played'],'Batsman Average': TopBatsman['average'], 'batting_strike_rate': TopBatsman.batting_strike_rate, 
            'nan': TopBatsman.nan,'hit wicket':TopBatsman['hit wicket'], 'retired hurt':TopBatsman['retired hurt'], 'obstructing the field':TopBatsman['obstructing the field'],
            '0s': TopBatsman['0s'], '1s': TopBatsman['1s'],'2s': TopBatsman['2s'],'3s': TopBatsman['3s'],'4s': TopBatsman['4s'],'5s': TopBatsman['5s'],'6s': TopBatsman['6s'],
            'Total number of 6s- Avg':TopBatsman['6s/match'],'Total number of 4s- Avg':TopBatsman['4s/match'] })
TopBatsman1 = TopBatsman1.reset_index()
TopBatsman1.rename(columns = {'index':'batsman'}, inplace = True)
TopBatsman1

labels_4s = TopBatsman1.iloc[:,5]
labels_6s = TopBatsman1.iloc[:,7]
"""import matplotlib.pyplot as plt
import numpy as np

data_to_plot = TopBatsman['6s/match']
data_to_plot1 = TopBatsman['4s/match']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 6), sharey=True)
axes[0, 0].boxplot(data_to_plot, labels=labels_4s, showmeans=True)
axes[0, 0].set_title('showmeans=True', fontsize=fs)

axes[0, 1].boxplot(data, labels=labels_6s, showmeans=True, meanline=True)
axes[0, 1].set_title('showmeans=True,\nmeanline=True', fontsize=fs)

for ax in axes.flatten():
    ax.set_yscale('log')
    ax.set_yticklabels([])

fig.subplots_adjust(hspace=0.4)
plt.show()
"""

"""fig, ax = plt.subplots(1,2, figsize=(9,4))

# matplotlib > 1.4
bp = ax[0].boxplot(data_to_plot, showmeans=True)
ax[0].set_title("Using showmeans")

#matpltolib < 1.4
bp = ax[1].boxplot(data_to_plot, positions=positions)
#means = [np.mean(data) for data in data_to_plot.T]
#ax[1].plot(positions, means, 'rs')
#ax[1].set_title("Plotting means manually")

plt.show()
"""
TopBatsman['dot_balls/total_balls'] = TopBatsman['0s']/TopBatsman['balls']  
TopBatsman['dot_balls/total_balls']
max_runs = df.groupby(['batsman'])['batsman_runs'].sum()
max_runs.sort_values(ascending=False,inplace=True)
max_runs[:10].plot(kind='bar')
#Favoutite Umpires
plt.subplots(figsize=(10,6))
ump=pd.concat([data['umpire1'],data['umpire2']]) 
ax=ump.value_counts().head(10).plot.bar(width=0.8,color='Y')
for p in ax.patches:
    ax.annotate(format(p.get_height()), (p.get_x()+0.15, p.get_height()+0.25))
plt.show()
#Season's Winner
season_winner = data.drop_duplicates(subset=['season'], keep='last')[['season', 'winner']].reset_index(drop=True)
season_winner
#Number of Wickets taken:
bowler_wickets = df.groupby('bowler').aggregate({'ball': 'count', 'total_runs': 'sum', 'player_dismissed' : 'count'})
bowler_wickets.columns = ['runs','balls','wickets']
TopBowlerBywickets = bowler_wickets.sort_values(['wickets'], ascending=False)[:20]
TopBowlerBywickets
TopBowlerBywickets['strikerate'] = TopBowlerBywickets['runs']/TopBowlerBywickets['wickets']
TopBowlerBywickets = TopBowlerBywickets.sort_values(['strikerate'], ascending=False)[:20]
TopBowlerBywickets

