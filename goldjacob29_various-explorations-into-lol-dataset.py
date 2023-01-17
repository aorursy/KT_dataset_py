import numpy as np
import pandas as pd
from pandas import DataFrame
import pprint as pp
import matplotlib.pyplot as plt
from ast import literal_eval

import os
print(os.listdir("../input"))
df_columns = pd.read_csv('../input/_columns.csv',sep=',')
df_matchinfo = pd.read_csv('../input/matchinfo.csv',sep=',')
df_raw = pd.read_csv('../input/LeagueofLegends.csv',sep=',')
df_kills = pd.read_csv('../input/kills.csv', sep=',')
df_monsters = pd.read_csv('../input/monsters.csv', sep=',')
df_raw.head(1)

# find the most banned champions against TSM in TSM losses
tsm_blue_losses = df_raw.loc[(df_raw['blueTeamTag'] == 'TSM') & (df_raw['bResult'] == 0)]
tsm_red_losses = df_raw.loc[(df_raw['redTeamTag'] == 'TSM') & (df_raw['rResult'] == 0)]

blue_loss_bans_against = tsm_blue_losses['redBans']
red_loss_bans_against = tsm_red_losses['blueBans']

bans_against = pd.concat([blue_loss_bans_against, red_loss_bans_against])

# weird parsing here due to bans being a list
def parse_bans(bans_against):
    bans_dict = defaultdict(int)
    for ban in bans_against:
        bans = ban.split(',')
        ban_1 = bans[0][2:-1]
        ban_2 = bans[1][2:-1]
        end_index = (-2 if bans[2][-1] == ']' else -1)
        ban_3 = bans[2][2:end_index]
        parsed_bans = [ban_1, ban_2, ban_3]
        for ban in parsed_bans:
            bans_dict[ban] += 1
    return bans_dict

blue_loss_bans_dict = parse_bans(blue_loss_bans_against)
red_loss_bans_dict = parse_bans(red_loss_bans_against)
loss_bans_dict = parse_bans(bans_against)

blue_champs_banned = Counter(blue_loss_bans_dict)
red_champs_banned = Counter(red_loss_bans_dict)
all_champs_banned = Counter(loss_bans_dict)

blue_loss_bans_dict = sorted(blue_loss_bans_dict.items(), key=lambda x: x[1], reverse=True)
red_loss_bans_dict = sorted(red_loss_bans_dict.items(), key=lambda x: x[1], reverse=True)
loss_bans_dict = sorted(loss_bans_dict.items(), key=lambda x: x[1], reverse=True)
# print('\n---most banned champions against TSM in TSM losses on BLUE side---')
# print(blue_loss_bans_dict)
#
# print('\n---most banned champions against TSM in TSM losses on RED side---')
# print(red_loss_bans_dict)
#
# print('\n---most banned champions against TSM in TSM losses on BOTH sides---')
# print(loss_bans_dict)

# find the most played champions by TSM in TSM wins

tsm_blue_wins = df_raw.loc[(df_raw['blueTeamTag'] == 'TSM') & (df_raw['bResult'] == 1)]
tsm_red_wins = df_raw.loc[(df_raw['redTeamTag'] == 'TSM') & (df_raw['rResult'] == 1)]

blue_champs_played = Counter()
red_champs_played = Counter()

blue_tops = Counter(tsm_blue_wins['blueTopChamp'].value_counts().to_dict())
blue_junglers = Counter(tsm_blue_wins['blueJungleChamp'].value_counts().to_dict())
blue_mids = Counter(tsm_blue_wins['blueMiddleChamp'].value_counts().to_dict())
blue_adcs = Counter(tsm_blue_wins['blueADCChamp'].value_counts().to_dict())
blue_supports = Counter(tsm_blue_wins['blueSupportChamp'].value_counts().to_dict())

blue_champs_played = blue_champs_played + blue_tops + blue_junglers + blue_mids + blue_adcs + blue_supports

red_tops = Counter(tsm_blue_wins['redTopChamp'].value_counts().to_dict())
red_junglers = Counter(tsm_blue_wins['redJungleChamp'].value_counts().to_dict())
red_mids = Counter(tsm_blue_wins['redMiddleChamp'].value_counts().to_dict())
red_adcs = Counter(tsm_blue_wins['redADCChamp'].value_counts().to_dict())
red_supports = Counter(tsm_blue_wins['redSupportChamp'].value_counts().to_dict())

red_champs_played = red_champs_played + red_tops + red_junglers + red_mids + red_adcs + red_supports

blue_champs_played_list = sorted(blue_champs_played.items(), key=lambda x: x[1], reverse=True)
red_champs_played_list = sorted(red_champs_played.items(), key=lambda x: x[1], reverse=True)
# print('\n---most played champions by TSM in TSM wins on BLUE side---')
# print(blue_champs_played_list)

# print('\n---most played champions by TSM in TSM wins on RED side---')
# print(red_champs_played_list)

blue_champs = blue_champs_played + blue_champs_banned
red_champs = red_champs_played + red_champs_banned

blue_champs = sorted(blue_champs.items(), key=lambda x: x[1], reverse=True)
red_champs = sorted(red_champs.items(), key=lambda x: x[1], reverse=True)
# print('\n---bans + played on BLUE side---')
# print(blue_champs)

# print('\n---bans + played on RED side---')
# print(red_champs)

# might also want to consider all champions played by them in wins and losses.
# they seem to think they are good and maybe that's worth something

sides = ['blue', 'red']
roles = ['Top', 'Jungle', 'Middle', 'ADC', 'Support']

champ_roles = []
for role in roles:
    champ_roles.append(role + 'Champ')

champs_played = defaultdict(int)
for index, game in df_raw.iterrows():
    for side in sides:
        if game[side + 'TeamTag'] == 'TSM':
            for champ_role in champ_roles:
                champs_played[game[side + champ_role]] +=1
champs_played = sorted(champs_played.items(), key=lambda x: x[1], reverse=True)
# champs_played
# stretches or shrinks v to be of size n
def procrustify(v, n):
    w = {}
    f = float(len(v))/n
    for i, d in enumerate(v):
        j = int(i/f)
        if not j in w:
            w[j] = []
        w[j].append(d)
    x = []
    nx = 0
    for k, v in w.items():
        while len(x) < k:
            x.append(x[-1])
        x.append(np.mean(v))
    while len(x) < n:
        x.append(x[-1])
    return np.array(x)

def test_procrustify():
    x = list(range(53))
    print('x=', x)
    y = procrustify(x, 10)
    print('y=', y)

sides = ['blue', 'red']
roles = ['Top', 'Jungle', 'Middle', 'ADC', 'Support']

positions = []
gold_vals = []
position_to_role = {}

for side in sides:
    for role in roles:
        sideprole = side + role
        positions.append(sideprole)
        position_to_role[sideprole] = role
        gold_vals.append('gold' + sideprole)
reqd_cols = positions + gold_vals
# restricting to NA only
df_playergold = df_raw.loc[df_raw['League'] == 'NALCS'][reqd_cols]

for col in reqd_cols:
    if 'gold' in col:
        df_playergold[col] = np.array(df_playergold[col].apply(literal_eval))
    else:
        df_playergold[col] = df_playergold[col].str.lower()

# just to make sure I don't add new keys to the dictionary...
NUM_GAMES = 'num_games'
GOLD_DIFFS = 'gold_diffs'
NORMALIZED_GOLD_DIFFS = 'norm_gold_diffs'
AVERAGE_GOLD_DIFF = 'avg_gold_diff'
ROLE = 'role'

# restricting to NA only
players = {}
for i, game in df_playergold.iterrows(): # assuming here that matchinfo and raw agree on all games
    for position in positions:
        name = game[position]
        d = players.get(name)
        if not d:
            d = {
                NUM_GAMES : 0,
                GOLD_DIFFS : [],
                NORMALIZED_GOLD_DIFFS : [],
                ROLE : position_to_role[position],
             }
            players[name] = d
        d[NUM_GAMES] += 1
        # assert position_to_role[position] == d[ROLE], "player {player} moved role! (<{old}> != <{new}>)".format(
        #         player=name, old=d[ROLE], new=position_to_role[position])


# players with at least 50 games; could change this number
# all_players_50_plus = {k: v for k, v in players.items() if v >= 50}
# all_players_50_plus
TIME_SLICES = 10

for i, game in df_playergold.iterrows():
    for role in roles:
        gold_diff = np.array(game[('goldblue' + role)]) - np.array(game[('goldred' + role)])
        blue_player, red_player = game['blue' + role], game['red' + role]
        players[blue_player][GOLD_DIFFS].append(gold_diff)
        players[red_player][GOLD_DIFFS].append(gold_diff * -1)
        players[blue_player][NORMALIZED_GOLD_DIFFS].append(procrustify(gold_diff, TIME_SLICES))
        players[red_player][NORMALIZED_GOLD_DIFFS].append(procrustify(gold_diff * -1, TIME_SLICES))

xx = 0
for name, player in players.items():
    player[AVERAGE_GOLD_DIFF] = sum(player[NORMALIZED_GOLD_DIFFS])/len(player[NORMALIZED_GOLD_DIFFS])
    if xx < 10:
        xx += 1
        print(name, ':', player[NUM_GAMES], ':', player[AVERAGE_GOLD_DIFF])
        
MIN_GAMES = 100
        
legend = []
for role in roles:
    for name, player in players.items():
        if player[NUM_GAMES] >= MIN_GAMES:
            if player[ROLE] == role:
                xvals = range(TIME_SLICES)
                yvals = player[AVERAGE_GOLD_DIFF]
                plt.plot(xvals, yvals)
                legend.append(name)
    plt.legend(legend, loc='upper left')
    plt.title(role + ' in ' +  str(TIME_SLICES) + ' Slices' )
    plt.show()
    plt.clf()
    legend = []

df_monsters.head(5)
df_raw.head(1)

#first bloods and first dragons
reqd_cols = ['bKills', 'bDragons', 'rKills', 'rDragons']
df_dragons_kills = df_raw[reqd_cols].copy()

for col in reqd_cols:
    df_dragons_kills[col] = df_dragons_kills[col].apply(literal_eval)

df_dragons_kills.head(1)
for game in df_dragons_kills.iterrows():
    print(game)
# #     print(first_blue_kill)
