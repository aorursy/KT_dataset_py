import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import init_notebook_mode, iplot
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

plt.style.use('bmh')
%matplotlib inline
plt.rcParams['figure.dpi'] = 100
init_notebook_mode(connected=True) 

data = pd.read_csv('../input/worlds2018.csv')

def comma_to_point(x):
    return x.replace(',','.')
data['gamelength'] = data['gamelength'].apply(comma_to_point).astype('float64')
data['fttime'] = data['fttime'].apply(comma_to_point).astype('float64')

data.head(12)
player_names = data.player.unique()

# Initialize the Dataframe
players = pd.DataFrame(data[(data.player == 'Licorice')].mean())
players = players.transpose()
players.rename(index={0:'Licorice'}, inplace=True)

# Append each player
for player in player_names:
    if player != 'Team' and player != 'Licorice':
        p = pd.DataFrame((data[(data.player == player)].mean())).transpose()
        p.rename(index={0: player}, inplace=True)
        players = pd.concat([players, p], sort=False)
        
# Establish players mean max min median might be useful later
player_mean = pd.DataFrame(players.mean()).transpose()
player_max = pd.DataFrame(players.max()).transpose()
player_min = pd.DataFrame(players.min()).transpose()
player_median = pd.DataFrame(players.median()).transpose()

# Wards, dmg, KDA, Kill participation
# Gold diff at 15, Total damage, Gold spread
labels = ['wards','dmgtochamps','totalgold','k','d','a','gdat15']

# update player positions as they had been lost
players.insert(len(players.columns), column='position',value='ADC')
players.insert(len(players.columns), column='team',value='Cloud9')
for player in player_names:
    if player != 'Team':
        players.loc[player, 'position'] = data[data.player == player].position.values[0]
        players.loc[player, 'team'] = data[data.player == player].team.values[0]


team_names = data.team.unique()
teams = pd.DataFrame(data[(data.player == 'Team') & (data.team == 'Cloud9')].mean())
teams = teams.transpose()
teams.rename(index={0:'Cloud9'}, inplace=True)

for team in team_names:
    if team != 'Cloud9':
        p = pd.DataFrame((data[(data.team == team) & (data.player == 'Team')].mean())).transpose()
        p.rename(index={0: team}, inplace=True)
        teams = pd.concat([teams, p], sort=False)
# Used for data visualization later
def simple_radar_graph(player_name, color='blue', players=players, role='ADC'):
    x = pd.DataFrame(players.loc[player_name, labels]).transpose()
    player_max = pd.DataFrame(players[player.position == role].max()).transpose()
    
    data = [go.Scatterpolar(
      r = [(x['wards'].values[0] / player_max['wards'].values[0]) * 100,(x['dmgtochamps'].values[0] / player_max['dmgtochamps'].values[0]) * 100,100 * (x['totalgold'].values[0] / player_max['totalgold'].values[0]),100 * (x['k'].values[0] / player_max['k'].values[0]),100 * (x['d'].values[0] / player_max['d'].values[0]),100 * (x['a'].values[0] / player_max['a'].values[0]),100 * (x["gdat15"].values[0] / player_max['gdat15'].values[0]), 100 * (x["wards"].values[0] / player_max['wards'].values[0])],
      theta = ['Wards','Damages to champs', 'Total gold','Kills','Deaths','Assists','Opponent gold difference at 15m','Wards'],
      fill = 'toself',
         line =  dict(
                color = color
            )
    )]

    layout = go.Layout(
      polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, 100]
        )
      ),
      showlegend = False,
      title = "{}".format(player_name)
    )
    fig = go.Figure(data=data, layout=layout)
    

    return iplot(fig, filename = "Radar")

# used for data visualization later
def vs_radar_graph(player_name1, player_name2, color1='orange', color2='lightgray',title=None, players=players,role='ADC'):
    player_max = pd.DataFrame(players[players.position == role].max()).transpose()
    
    if not title:
         title='{} vs {}'.format(player_name1, player_name2)
    
    x1 = pd.DataFrame(players.loc[player_name1, labels]).transpose()
    x2 = pd.DataFrame(players.loc[player_name2, labels]).transpose()
    # MEDIAN : x3 = pd.DataFrame(players[players.position == role].median()).transpose()
    
    """
        MEDIAN
        go.Scatterpolar(
      name = player_name2,
      r = [(x3['wards'].values[0] / player_max['wards'].values[0]) * 100,(x3['dmgtochamps'].values[0] / player_max['dmgtochamps'].values[0]) * 100,100 * (x3['totalgold'].values[0] / player_max['totalgold'].values[0]),100 * (x3['k'].values[0] / player_max['k'].values[0]),100 * (x3['d'].values[0] / player_max['d'].values[0]),100 * (x3['a'].values[0] / player_max['a'].values[0]),100 * (x3["wards"].values[0] / player_max['wards'].values[0])],
      theta = ['Wards','Damages to champs', 'Total gold','Kills','Deaths','Assists','Wards'],
      fill = 'toself',
         line =  dict(
                color = 'black'
            )
        ),
        
    add as 1st element in array below if you want
    """
    
    data = [go.Scatterpolar(
      name = player_name1,
      r = [(x1['wards'].values[0] / player_max['wards'].values[0]) * 100,(x1['dmgtochamps'].values[0] / player_max['dmgtochamps'].values[0]) * 100,100 * (x1['totalgold'].values[0] / player_max['totalgold'].values[0]),100 * (x1['k'].values[0] / player_max['k'].values[0]),100 * (x1['d'].values[0] / player_max['d'].values[0]),100 * (x1['a'].values[0] / player_max['a'].values[0]),100 * (x1["gdat15"].values[0] / player_max['gdat15'].values[0]),100 * (x1["wards"].values[0] / player_max['wards'].values[0])],
      theta = ['Wards','Damages to champs', 'Total gold','Kills','Deaths','Assists','Opponent gold difference at 15m','Wards'],
      fill = 'toself',
         line =  dict(
                color = color1
            )
    ),
           
    go.Scatterpolar(
      name = player_name2,
      r = [(x2['wards'].values[0] / player_max['wards'].values[0]) * 100,(x2['dmgtochamps'].values[0] / player_max['dmgtochamps'].values[0]) * 100,100 * (x2['totalgold'].values[0] / player_max['totalgold'].values[0]),100 * (x2['k'].values[0] / player_max['k'].values[0]),100 * (x2['d'].values[0] / player_max['d'].values[0]),100 * (x2['a'].values[0] / player_max['a'].values[0]),100 * (x2["gdat15"].values[0] / player_max['gdat15'].values[0]),100 * (x2["wards"].values[0] / player_max['wards'].values[0])],
      theta = ['Wards','Damages to champs', 'Total gold','Kills','Deaths','Assists','Opponent gold difference at 15m','Wards'],
      fill = 'toself',
         line =  dict(
                color = color2
            )
    )
           
    ]

    layout = go.Layout(
      polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, 100]
        )
      ),
      showlegend = False,
      title = "{}".format(title)
    )
    fig = go.Figure(data=data, layout=layout)
    

    return iplot(fig, filename = "Radar")

teams_labels = ['fb','gamelength', 'k', 'd', 'dmgtochamps','totalgold','fbaron','herald','ft','firstmidouter','firsttothreetowers','gdat15']
# used for data visualization later
def teamvs_radar_graph(team_name1, team_name2, color1='orange', color2='lightgray',title=None, teams=teams):
    team_max = pd.DataFrame(teams.max()).transpose()
    
    if not title:
         title='{} vs {}'.format(team_name1, team_name2)
    
    x1 = pd.DataFrame(teams.loc[team_name1, teams_labels]).transpose()
    x2 = pd.DataFrame(teams.loc[team_name2, teams_labels]).transpose()

    data = [go.Scatterpolar(
      name = team_name1,
      r = [(x1['fb'].values[0] / team_max['fb'].values[0]) * 100,(x1['gamelength'].values[0] / team_max['gamelength'].values[0]) * 100,(x1['k'].values[0] / team_max['k'].values[0]) * 100,(x1['d'].values[0] / team_max['d'].values[0]) * 100,(x1['dmgtochamps'].values[0] / team_max['dmgtochamps'].values[0]) * 100,100 * (x1['totalgold'].values[0] / team_max['totalgold'].values[0]),100 * (x1['fbaron'].values[0] / team_max['fbaron'].values[0]),100 * (x1['herald'].values[0] / team_max['herald'].values[0]),100 * (x1['ft'].values[0] / team_max['ft'].values[0]),100 * (x1["firstmidouter"].values[0] / team_max['firstmidouter'].values[0]),100 * (x1["firsttothreetowers"].values[0] / team_max['firsttothreetowers'].values[0]),100 * (x1["gdat15"].values[0] / team_max['gdat15'].values[0]),100 * (x1["fb"].values[0] / team_max['fb'].values[0])],
      theta =['First Blood', 'Game length', 'Kills', 'Deaths', 'Damages to champs','Total gold','First Baron','Herald','First tower','First mid outer tower','First to three towers','Gold difference at 15', 'First Blood'],
      fill = 'toself',
         line =  dict(
                color = color1
            )
    ),
           
    go.Scatterpolar(
      name = team_name2,
      r = [(x2['fb'].values[0] / team_max['fb'].values[0]) * 100,(x2['gamelength'].values[0] / team_max['gamelength'].values[0]) * 100,(x2['k'].values[0] / team_max['k'].values[0]) * 100, (x2['d'].values[0] / team_max['d'].values[0]) * 100,(x2['dmgtochamps'].values[0] / team_max['dmgtochamps'].values[0]) * 100,100 * (x2['totalgold'].values[0] / team_max['totalgold'].values[0]),100 * (x2['fbaron'].values[0] / team_max['fbaron'].values[0]),100 * (x2['herald'].values[0] / team_max['herald'].values[0]),100 * (x2['ft'].values[0] / team_max['ft'].values[0]),100 * (x2["firstmidouter"].values[0] / team_max['firstmidouter'].values[0]),100 * (x2["firsttothreetowers"].values[0] / team_max['firsttothreetowers'].values[0]),100 * (x2["gdat15"].values[0] / team_max['gdat15'].values[0]),100 * (x2["fb"].values[0] / team_max['fb'].values[0])],
      theta =['First Blood', 'Game length', 'Kills', 'Deaths', 'Damages to champs','Total gold','First Baron','Herald','First tower','First mid outer tower','First to three towers','Gold difference at 15', 'First Blood'],
      fill = 'toself',
         line =  dict(
                color = color2
            )
    )
           
    ]

    layout = go.Layout(
      polar = dict(
        radialaxis = dict(
          visible = True,
          range = [0, 100]
        )
      ),
      showlegend = False,
      title = "{}".format(title)
    )
    fig = go.Figure(data=data, layout=layout)
    

    return iplot(fig, filename = "Radar")
players_corr = players

# to avoid running the kernel everytime
if 'gameid' in players_corr.columns.tolist():
    # doubles/triples/quadras/pentas/okpm have suffered damages with the .mean() for unknown reasons,
    # elementals, airdrakes, ... are chosen with randomness so they're not significant
    # heraldtime barontime are not contained in players variables
    # time var : make no sense in corr
    players_corr = players_corr.drop(['doubles','triples','quadras','pentas','gameid','playerid','firedrakes','elders','elementals','waterdrakes','earthdrakes','airdrakes','heraldtime','fbarontime','okpm', 'fttime', 'gamelength'], axis=1)

corr = players_corr.corr()
f, ax = plt.subplots(figsize=(24, 16))
heatmap = sns.heatmap(corr)
graph_data = pd.concat([players['gdat10'], players['result']], axis=1)
graph_data.plot.scatter(x='gdat10', y='result');
plt.title('Gold Difference to opponent at 10 minutes impact on winrate (%)')
plt.show()
graph_data = pd.concat([players['xpdat10'], players['result']], axis=1)
graph_data.plot.scatter(x='xpdat10', y='result');
plt.title('XP Difference to opponent at 10 minutes impact on winrate (%)')
plt.show()
graph_data = pd.concat([teams['fb'], teams['result']], axis=1)
graph_data.plot.scatter(x='fb', y='result');
plt.title('First blood impact on winrate (%)')
plt.show()
graph_data = pd.concat([teams['fd'], teams['result']], axis=1)
graph_data.plot.scatter(x='fd', y='result');
plt.title('First dragon of the game impact on winrate (%)')
plt.show()
graph_data = pd.concat([teams['fbaron'], teams['result']], axis=1)
graph_data.plot.scatter(x='fbaron', y='result');
plt.title('First baron of the game impact on winrate (%)')
plt.show()
graph_data = pd.concat([teams['herald'], teams['result']], axis=1)
graph_data.plot.scatter(x='herald', y='result');
plt.title('Herald impact on winrate (%)')
plt.show()
graph_data = pd.concat([teams['ft'], teams['result']], axis=1)
graph_data.plot.scatter(x='ft', y='result');
plt.title('First tower destruction impact on winrate (%)')
plt.show()
graph_data = pd.concat([teams['firstmidouter'], teams['result']], axis=1)
graph_data.plot.scatter(x='firstmidouter', y='result');
plt.title('First outer mid tower destruction impact on winrate (%)')
plt.show()
graph_data = pd.concat([teams['firsttothreetowers'], teams['result']], axis=1)
graph_data.plot.scatter(x='firsttothreetowers', y='result');
plt.title('First to three destroyed towers impact on winrate (%)')
plt.show()
vs_radar_graph('TheShy','Bwipo', title='TOPs | (orange) FNC Bwipo vs IG TheShy (lightgray)', role='Top')
vs_radar_graph('Broxah','Ning', title='Junglers | (orange) FNC Broxah vs IG Ning (MVP) (lightgray)', role='Jungle')
vs_radar_graph('Rekkles','RooKie', title='MIDs | (orange) FNC Caps vs IG RooKie (lightgray)', role='Middle')
vs_radar_graph('Rekkles','JackeyLove', title='ADCs | (orange) FNC Rekkles vs IG JackeyLove (lightgray)', role='ADC')
vs_radar_graph('Hylissang','Baolan', title='Supports | (orange) FNC Hylissang vs IG Baolan (lightgray)', role='Support')
teamvs_radar_graph('Fnatic', 'Invictus Gaming', color1='orange', color2='lightgray',title='(orange) Fnatic vs Invictus Gaming (lightgray)', teams=teams)
team_name = 'Invictus Gaming'
print('{} average game length on win : {} minutes'.format(team_name,data[(data.player == 'Team') & (data.team == team_name) & (data.result == 1)]['gamelength'].mean()))
print('{} average game length on lose : {} minutes'.format(team_name,data[(data.player == 'Team') & (data.team == team_name) & (data.result == 0)]['gamelength'].mean()))

team_name = 'Fnatic'
print('\n{} average game length on win : {} minutes'.format(team_name,data[(data.player == 'Team') & (data.team == team_name) & (data.result == 1)]['gamelength'].mean()))
print('{} average game length on lose : {} minutes'.format(team_name,data[(data.player == 'Team') & (data.team == team_name) & (data.result == 0)]['gamelength'].mean()))
teamvs_radar_graph('Fnatic', 'G2 Esports', color1='orange', color2='gray',title='(orange) Fnatic vs G2 Esports (gray)', teams=teams)
teamvs_radar_graph('Invictus Gaming', 'Royal Never Give Up', color1='lightgray', color2='brown',title='(lightgray) Invictus Gaming vs Royal Never Give Up (brown)', teams=teams)
graph_data = data[(data.player != 'Team')]
print('Number of different champions overall : {}'.format(len(graph_data['champion'].unique())))
sns.set(rc={'figure.figsize':(11.7,8.27)})

graph_data = data[(data.player != 'Team') & (data.position == 'Top')]
sns.countplot(y="champion",data=graph_data, order=graph_data['champion'].value_counts().index)
plt.title('Champion pick rate in Toplane')
plt.show()

print('Number of different champions in Toplane : {}'.format(len(graph_data['champion'].unique())))
graph_data = data[(data.player != 'Team') & (data.position == 'Jungle')]
sns.countplot(y="champion",data=graph_data, order=graph_data['champion'].value_counts().index)
plt.title('Champion pick rate in Jungle')
plt.show()

print('Number of different champions in Jungle : {}'.format(len(graph_data['champion'].unique())))
graph_data = data[(data.player != 'Team') & (data.position == 'Middle')]
sns.countplot(y="champion",data=graph_data, order=graph_data['champion'].value_counts().index)
plt.title('Champion pick rate in Middle')
plt.show()

print('Number of different champions in Middle : {}'.format(len(graph_data['champion'].unique())))
graph_data = data[(data.player != 'Team') & (data.position == 'ADC')]
sns.countplot(y="champion",data=graph_data, order=graph_data['champion'].value_counts().index)
plt.title('Champion pick rate in AD Carry')
plt.show()

print('Number of different champions in AD Carry : {}'.format(len(graph_data['champion'].unique())))
graph_data = data[(data.player != 'Team') & (data.position == 'Support')]
sns.countplot(y="champion",data=graph_data, order=graph_data['champion'].value_counts().index)
plt.title('Champion pick rate in Support')
plt.show()

print('Number of different champions in Support : {}'.format(len(graph_data['champion'].unique())))
ban = data[(data.player == 'Team')]['ban1'].append([data[(data.player == 'Team')]['ban2'], data[(data.player == 'Team')]['ban3'], data[(data.player == 'Team')]['ban4'], data[(data.player == 'Team')]['ban5']])

print('Number of different champions banned : {}'.format(len(ban.unique())))

fig = plt.figure(figsize=(15,15))
fig.add_subplot(1,1,1)
sns.countplot(y=ban,data=ban, order=ban.value_counts().index)
plt.title('Bans distribution')
plt.show()
print('Aatrox winrate : {}%'.format((data[(data.champion == 'Aatrox')]['result'].mean() * 100)))
print('Akali winrate : {}%'.format((data[(data.champion == 'Akali')]['result'].mean() * 100)))
print('Urgot winrate : {}%'.format((data[(data.champion == 'Urgot')]['result'].mean() * 100)))
sns.countplot(data[(data.player == 'Team') & (data.result == 1)]['side'], palette=['blue','red'])
plt.title('Wins per side')
plt.show()
sns.distplot(data[(data.player == 'Team')]['gamelength'])
plt.title('Game length distribution')
plt.show()
sns.distplot(data[(data.player == 'Team')]['fttime'])
plt.title('First tower destroyed time distribution')
plt.show()
graph_data = players
graph_data['mkills'] = graph_data['monsterkillsownjungle'].add(graph_data['monsterkillsenemyjungle'])
graph_data = graph_data.sort_values(by=['mkills'])

monsterkillsownjungle = graph_data[(graph_data.position == 'Jungle')]['monsterkillsownjungle'].tolist()
monsterkillsenemyjungle = graph_data[(graph_data.position == 'Jungle')]['monsterkillsenemyjungle'].tolist()
ind =  np.arange(len(graph_data[(graph_data.position == 'Jungle')]))
width = 0.9

p1 = plt.bar(ind, monsterkillsownjungle, width, yerr=None, color='lightgray')
p2 = plt.bar(ind, monsterkillsenemyjungle, width,
             bottom=monsterkillsownjungle, yerr=None, color='green')

plt.ylabel('Average jungle monster kills')
plt.title('Average jungle monster kills distribution')
plt.xticks(ind, graph_data[(graph_data.position == 'Jungle')].index.tolist(), rotation=90)
plt.yticks(np.arange(0, 121, 10))
plt.legend((p1[0], p2[0]), ('Average monster kills own jungle', 'Average monster kills enemy jungle'))

plt.show()
graph_data = pd.DataFrame()
graph_data.insert(len(graph_data.columns), column='gold_top',value=0)
graph_data.insert(len(graph_data.columns), column='gold_jungle',value=0)
graph_data.insert(len(graph_data.columns), column='gold_middle',value=0)
graph_data.insert(len(graph_data.columns), column='gold_adc',value=0)
graph_data.insert(len(graph_data.columns), column='gold_support',value=0)
graph_data.insert(len(graph_data.columns), column='gold_all',value=0)

for team in team_names:
    gold_top = data[(data.position == 'Top') & (data.team == team)]['totalgold'].mean()
    gold_jungle = data[(data.position == 'Jungle') & (data.team == team)]['totalgold'].mean()
    gold_middle = data[(data.position == 'Middle') & (data.team == team)]['totalgold'].mean()
    gold_adc = data[(data.position == 'ADC') & (data.team == team)]['totalgold'].mean()
    gold_support = data[(data.position == 'Support') & (data.team == team)]['totalgold'].mean()
    gold_all = data[(data.position == 'Top') & (data.team == team)]['totalgold'].mean() + data[(data.position == 'Jungle') & (data.team == team)]['totalgold'].mean() + data[(data.position == 'Middle') & (data.team == team)]['totalgold'].mean() + data[(data.position == 'ADC') & (data.team == team)]['totalgold'].mean() + data[(data.position == 'Support') & (data.team == team)]['totalgold'].mean()

    graph_data.loc[team] = [gold_top, gold_jungle, gold_middle, gold_adc, gold_support, gold_all]

graph_data = graph_data.sort_values(by=['gold_all'])

bottom1 = graph_data['gold_support']
bottom2 = bottom1.add(graph_data['gold_adc'])
bottom3 = bottom2.add(graph_data['gold_middle'])
bottom4 = bottom3.add(graph_data['gold_jungle'])

support = graph_data['gold_support'].tolist()
adc = graph_data['gold_adc'].tolist()
middle = graph_data['gold_middle'].tolist()
jungle = graph_data['gold_jungle'].tolist()
top = graph_data['gold_top'].tolist()

ind =  np.arange(len(graph_data))
width = 0.9

p1 = plt.bar(ind, support, width, yerr=None, color='lightgray')
p2 = plt.bar(ind, adc, width, bottom=bottom1.tolist(), yerr=None, color='gold')
p3 = plt.bar(ind, middle, width, bottom=bottom2.tolist(), yerr=None, color='royalblue')
p4 = plt.bar(ind, jungle, width, bottom=bottom3.tolist(), yerr=None, color='darkseagreen')
p5 = plt.bar(ind, top, width, bottom=bottom4.tolist(), yerr=None, color='indianred')

plt.ylabel('Team average total gold')
plt.title('Average team total gold distribution per role')
plt.xticks(ind, graph_data.index.tolist(), rotation=90)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Support', 'ADC', 'Mid', 'Jungle', 'Top'))

plt.show()
graph_data = pd.DataFrame()
graph_data.insert(len(graph_data.columns), column='kills_top',value=0)
graph_data.insert(len(graph_data.columns), column='kills_jungle',value=0)
graph_data.insert(len(graph_data.columns), column='kills_middle',value=0)
graph_data.insert(len(graph_data.columns), column='kills_adc',value=0)
graph_data.insert(len(graph_data.columns), column='kills_support',value=0)
graph_data.insert(len(graph_data.columns), column='kills_all',value=0)

for team in team_names:
    kills_top = data[(data.position == 'Top') & (data.team == team)]['k'].mean()
    kills_jungle = data[(data.position == 'Jungle') & (data.team == team)]['k'].mean()
    kills_middle = data[(data.position == 'Middle') & (data.team == team)]['k'].mean()
    kills_adc = data[(data.position == 'ADC') & (data.team == team)]['k'].mean()
    kills_support = data[(data.position == 'Support') & (data.team == team)]['k'].mean()
    kills_all = data[(data.position == 'Top') & (data.team == team)]['k'].mean() + data[(data.position == 'Jungle') & (data.team == team)]['k'].mean() + data[(data.position == 'Middle') & (data.team == team)]['k'].mean() + data[(data.position == 'ADC') & (data.team == team)]['k'].mean() + data[(data.position == 'Support') & (data.team == team)]['k'].mean()

    graph_data.loc[team] = [kills_top, kills_jungle, kills_middle, kills_adc, kills_support, kills_all]

graph_data = graph_data.sort_values(by=['kills_all'])

bottom1 = graph_data['kills_support']
bottom2 = bottom1.add(graph_data['kills_adc'])
bottom3 = bottom2.add(graph_data['kills_middle'])
bottom4 = bottom3.add(graph_data['kills_jungle'])

support = graph_data['kills_support'].tolist()
adc = graph_data['kills_adc'].tolist()
middle = graph_data['kills_middle'].tolist()
jungle = graph_data['kills_jungle'].tolist()
top = graph_data['kills_top'].tolist()

ind =  np.arange(len(graph_data))
width = 0.9

p1 = plt.bar(ind, support, width, yerr=None, color='lightgray')
p2 = plt.bar(ind, adc, width, bottom=bottom1.tolist(), yerr=None, color='gold')
p3 = plt.bar(ind, middle, width, bottom=bottom2.tolist(), yerr=None, color='royalblue')
p4 = plt.bar(ind, jungle, width, bottom=bottom3.tolist(), yerr=None, color='darkseagreen')
p5 = plt.bar(ind, top, width, bottom=bottom4.tolist(), yerr=None, color='indianred')

plt.ylabel('Team average total kills')
plt.title('Average team total kills distribution per role')
plt.xticks(ind, graph_data.index.tolist(), rotation=90)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Support', 'ADC', 'Mid', 'Jungle', 'Top'))
plt.show()