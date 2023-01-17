import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/nflplaybyplay2015.csv")
import warnings
warnings.filterwarnings('ignore')
df.columns.values
gb_o = df[df['posteam']=='GB']
gb_d = df[df['DefensiveTeam']=='GB']
game = {}
game_ids = gb_o.GameID.unique()
for index,id in enumerate(game_ids):
    game[index+1] = id
game[1]
game_s = pd.Series(game) 
game_1 = gb_o[gb_o['GameID']==game[1]]    
completions = game_1[game_1['PassOutcome'] == 'Complete']
completions['Yards.Gained']
plt.plot(completions['Yards.Gained'])
plt.hist(completions['Yards.Gained'])
pass_yd = []
rush_yd = []
game_ = []
for i in range(16):
    game_.append(i+1)
    g = gb_o[gb_o['GameID']==game[i+1]] 
    completions = g[g['PassOutcome'] == 'Complete']
    rushes = g[g['RushAttempt'] == 1]
    pass_yd.append(completions['Yards.Gained'].sum())
    rush_yd.append(rushes['Yards.Gained'].sum())
plt.plot(game_, pass_yd, label='pass')
plt.plot(game_, rush_yd, label='rush')
plt.legend()
gb_dn = gb_o[gb_o['GameID']==game[7]] 
# The final score
gb_dn[['PosTeamScore', 'DefTeamScore']].tail()
