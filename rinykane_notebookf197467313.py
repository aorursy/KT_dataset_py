import sqlite3

import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from math import sqrt
COUNTRY_ID = 7809 #Germany

LEAGUE_ID = 7809  #Bundes liga

SEASON = "2015/2016"

DEBUG = 0
# Create your connection.

cnx = sqlite3.connect('database.sqlite')

df = pd.read_sql_query("SELECT * FROM Match", cnx)
# Data based on country, league and season

mask_c = df['country_id'] == COUNTRY_ID

mask_l = df['league_id'] == LEAGUE_ID

mask_s = df['season'] == SEASON

league_and_season = df[mask_c & mask_l & mask_s]
league_and_season.shape
league_and_season[:1]
len(df)
features = ['id', 'country_id', 'league_id', 'season', 'stage', 'date',

       'match_api_id', 'home_team_api_id', 'away_team_api_id','home_team_goal', 'away_team_goal',

#        'home_player_X1', 'home_player_X2', 'home_player_X3', 'home_player_X4', 'home_player_X5', 'home_player_X6', 'home_player_X7', 'home_player_X8', 'home_player_X9', 'home_player_X10', 'home_player_X11', 

#        'away_player_X1','away_player_X2','away_player_X3','away_player_X4','away_player_X5','away_player_X6','away_player_X7','away_player_X8','away_player_X9','away_player_X10','away_player_X11',

        'home_player_Y1', 'home_player_Y2', 'home_player_Y3', 'home_player_Y4', 'home_player_Y5', 'home_player_Y6', 'home_player_Y7', 'home_player_Y8', 'home_player_Y9', 'home_player_Y10', 'home_player_Y11', 

        'away_player_Y1','away_player_Y2','away_player_Y3','away_player_Y4','away_player_Y5','away_player_Y6','away_player_Y7','away_player_Y8','away_player_Y9','away_player_Y10','away_player_Y11',

        'home_player_1', 'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7', 'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 

        'away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6','away_player_7','away_player_8','away_player_9','away_player_10','away_player_11',

        'goal','shoton','shotoff','foulcommit','card','cross','corner','possession']
ls = league_and_season[features]

cards = ls['card']

len(cards)
goals = ls['goal']

len(goals)

#

# For future evaluation

#

#possession = ls['possession']

#possession[9950]

#cross = ls['cross']

#cross[9950]

#shoton = ls['shoton']

#shoton[shoton.str.contains('penal')]

import declxml as xml

# processor for extracting goals from XML information

processor_goal = xml.dictionary('goal', [

    xml.array(xml.dictionary('value', [

        xml.integer('player1',required=False, omit_empty=True),

        xml.integer('player2',required=False, omit_empty=True),

        xml.integer('elapsed'),

        xml.integer('team'),

        xml.string('subtype',required=False, default='Missing'),

        xml.integer('id'),

        xml.array(xml.dictionary('stats', [

            xml.integer('goals',required=False, omit_empty=True, default=0),

            xml.integer('owngoals',required=False, omit_empty=True, default=0)

        ],required=False))

    ]))

])



# processor for extracting cards from XML information

processor_card = xml.dictionary('card', [

    xml.array(xml.dictionary('value', [

        xml.integer('player1',required=False, omit_empty=True),

        xml.integer('elapsed'),

        xml.string('card_type',required=False, default='Missing')

    ]))

])

all_cards = []

empty_c = {'value': [{'elapsed': 0,'player1': 0,'card_type': 'Missing'}]}

for x_line in cards.values:

    try:

        # Avoid empty lines

        if len(x_line) > 8:

            all_cards.append(xml.parse_from_string(processor_card, x_line))

        else :

            all_cards.append(empty_c)

    except :

        print(x_line, len(x_line))



#    print('.')

print(len(all_cards))

if DEBUG :

    print(x_line)

#xml.parse_from_string(processor_goal, goals[9950])

all_goals = []

empty_g = {'value': [{'elapsed': 0,'id': 0,'player1': 0,'player2': 0,'team': 0,

                      'stats': [{'goals': 0, 'owngoals': 0}],'subtype': 'NULL'}]}

for x_line in goals.values:

    try:

        # Avoid empty lines

        if len(x_line) > 8:

            all_goals.append(xml.parse_from_string(processor_goal, x_line))

        else:

            all_goals.append(empty_g)

#            print(x_line)

    except :

        print(x_line, len(x_line))

len(all_goals)
# Goals

game_goals = all_goals[0]['value']

# verify size of tables

print(len(ls),len(all_goals))

if DEBUG :

    game_goals
def goal_status (game):

    """ provides information about number of 

        0 :goals - int 

        1 :makers - list 

        2 :assist - list 

        3 :own goals - int

        4 :own goal makers - list 

        5 :winning goal maker - int

    """

    goals = owng = 0

    teama = wgi = gd = i = 0

    wgmaker = 0

    makers = []

    assists = []

    ogmakers = []

    for n in game :

        if len(n['stats']) > 0:

            # Find winning goal makers

            if not teama :

                teama = n['team']





            # Own goal

            if n['stats'][0]['owngoals'] :

                owng = owng + 1

                ogmakers.append(n['player1'])

                if teama == n['team'] :

                    gd = gd + 1

                else :

                    gd = gd - 1

                if gd == 1 or gd == -1:

                    wgi = -1



            # Goals

            else :

                if (n['stats'][0]['goals']) :

                    goals = goals + n['stats'][0]['goals']

                    makers.append(n['player1'])

                    assists.append(n['player2'])

                    if teama == n['team'] :

                        gd = gd + 1

                    else :

                        gd = gd - 1



                    if gd == 1 or gd == -1:

                        wgi = i



                    i = i + 1



    if gd and wgi > -1 : 

        if wgi >= goals :

            print("Index too high :", wgi, goals, gd, owng)

        else :

            wgmaker = makers[wgi]

    return (goals, makers, assists, owng, ogmakers, wgmaker)

    

# for testing only

if DEBUG :

    m = goal_status(game_goals)

    gm = m[1]

    ga = m[2]

    pid = 116772

    gm.count(pid)



    if pid in gm :

        print (gm[gm.values == pid].count())

        

def card_status (game):

    """ provides information about number of 

        0 :ycards - int 

        1 :yholders - list 

        2 :y2cards - int 

        3 :y2holders - list 

        4 :rcards - int 

        5 :rholders - list

    """

    ycards = y2cards = rcards = 0

    yholders = []

    y2holders = []

    rholders = []

    for n in game :

        # Yellow card

        if n['card_type'] == 'y' :

            ycards = ycards + 1

            yholders.append(n['player1'])

        # 2nd Yellow card

        if n['card_type'] == 'y2' :

            y2cards = ycards + 1

            y2holders.append(n['player1'])

        # Red cards

        if n['card_type'] == 'r' :

            rcards = rcards + 1

            rholders.append(n['player1'])



    return (ycards, yholders, y2cards, y2holders, rcards, rholders)

if DEBUG :

    all_cards[:2]
# Read from database and modify column

fifa_attr = pd.read_sql_query("SELECT player_api_id, overall_rating, date FROM Player_Attributes", cnx)

fifa_por = pd.DataFrame(fifa_attr)

fifa_por = fifa_por.rename(columns={"player_api_id": "id"})

#print("All players in DB: ",len(pl_a))



# Get additional Player information from database

fifa_player = pd.read_sql_query("SELECT player_api_id, player_name, birthday date FROM Player", cnx)

fifa_pname = pd.DataFrame(fifa_player)

fifa_pname = fifa_pname.rename(columns={"player_api_id": "id"})



# Use ratings only from 2 last year, use mean of multiple rating

fifa_por_lr = fifa_por[fifa_por['date'] > '2015-01-01']

del fifa_por_lr['date']

mask = ['home_player_Y1', 'home_player_Y2', 'home_player_Y3', 'home_player_Y4', 'home_player_Y5', 'home_player_Y6', 'home_player_Y7', 'home_player_Y8', 'home_player_Y9', 'home_player_Y10', 'home_player_Y11']

mask_a = ['away_player_Y1', 'away_player_Y2', 'away_player_Y3', 'away_player_Y4', 'away_player_Y5', 'away_player_Y6', 'away_player_Y7', 'away_player_Y8', 'away_player_Y9', 'away_player_Y10', 'away_player_Y11']

if DEBUG :

    ls[mask_a]
player_attr = ['games','goals','2goals','3goals','wgoal','ogoals','assists','csheet','wins','loses','loses3','ycards','y2cards','rcards','total']

player_pos = ['player_Y1','player_Y2','player_Y3','player_Y4','player_Y5','player_Y6','player_Y7','player_Y8','player_Y9','player_Y10','player_Y11']

def mark_positions(players, positions):

    """

        joins player information and position information 

    """

    d_pl = pd.DataFrame(players)

    d_po = pd.DataFrame(positions)

    d_po.index = d_po.index.str.replace('Y','')

    d_pp = pd.merge(d_pl, d_po, left_index=True, right_index=True)



    d_pp = d_pp.rename(index=str, columns={d_pp.columns[0]: "id", d_pp.columns[1]: "p_pos"})

    

    return(d_pp)



def map_positions(players_positions):

    """

        aplayer 

    """

    d_pp = pd.DataFrame(players_positions, columns=['id','p_pos'])

    

    if not len(d_pp) :

        print("Invalid input data : ", d_pp)

        return 0

    

    # First add required columns

    for pos in player_pos:

        d_pp[pos] = 0

        

    # Then map the positions

    for i in d_pp.index:

        pos = (int) (d_pp[d_pp.index == i]['p_pos'])

        p_text = 'player_Y' + str(pos)

        d_pp.loc[i, p_text] = 1



    return d_pp

y_pos = ['home_player_Y1','home_player_Y2','home_player_Y3','home_player_Y4','home_player_Y5','home_player_Y6','home_player_Y7','home_player_Y8','home_player_Y9','home_player_Y10','home_player_Y11','away_player_Y1','away_player_Y2','away_player_Y3','away_player_Y4','away_player_Y5','away_player_Y6','away_player_Y7','away_player_Y8','away_player_Y9','away_player_Y10','away_player_Y11']

player = ['home_player_1','home_player_2','home_player_3','home_player_4','home_player_5','home_player_6','home_player_7','home_player_8','home_player_9','home_player_10','home_player_11','away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6','away_player_7','away_player_8','away_player_9','away_player_10','away_player_11']



h_player = ['home_player_1','home_player_2','home_player_3','home_player_4','home_player_5','home_player_6','home_player_7','home_player_8','home_player_9','home_player_10','home_player_11']

a_player = ['away_player_1','away_player_2','away_player_3','away_player_4','away_player_5','away_player_6','away_player_7','away_player_8','away_player_9','away_player_10','away_player_11']





all_plpo = pd.DataFrame()

i = n = 0



# Analyse all games of the league/season

for item, frame in ls.iterrows():

    g_pos = frame[y_pos]

    g_pl = frame[player]

    

    # Find out player positions in a match

    pl_po = mark_positions(g_pl, g_pos)

    pl_po = map_positions(pl_po)

    

    for attr in player_attr:

        pl_po[attr] = 0



    htg = int(frame['home_team_goal'])

    atg = int(frame['away_team_goal'])

    hgdif = htg - atg

    hwin = 0

    hlose = 0

    clean_sheet = 0



    if hgdif > 0:

        hwin = 1

    elif hgdif < 0:

        hlose = 1

        

    if len(all_goals) <= n :

        print("Index error: Number of table rows: ", n)

    else :

        goals = all_goals[n]['value']

        game_g = goal_status(goals)



    gg = game_g[0] # number of game goals

    gm = game_g[1] # id list of goal makers

    ga = game_g[2] # id list of assists

    og = game_g[3] # number of own goals

    ogm = game_g[4] # id list of own goal makers

    wgm = game_g[5] # id of winning goal maker



    p_goals = 0

        

    # Mark goals

    for pid in gm :

        ci = pl_po[pl_po['id'] == pid].index

        p_goals = gm.count(pid)

        if p_goals > 2 :

            pl_po.loc[ci,'3goals'] = 1

        elif p_goals == 2 :

            pl_po.loc[ci,'2goals'] = 1

        elif p_goals == 1 :

            pl_po.loc[ci,'goals'] = pl_po.loc[ci,'goals'] + 1

        if pid == wgm :

            pl_po.loc[ci,'wgoal'] = 1



    # Mark assists

    for pid in ga :

        ci = pl_po[pl_po['id'] == pid].index

        pl_po.loc[ci,'assists'] = pl_po.loc[ci,'assists'] + 1



    # clean sheet

    if not atg :

        clean_sheet = 1



    # Own goals 

    for pid in ogm :

        ci = pl_po[pl_po['id'] == pid].index

        pl_po.loc[ci,'ogoals'] = pl_po.loc[ci,'ogoals'] + 1



    # Other statistic

    if hwin:

        pl_po.loc[h_player,'wins'] = 1

        pl_po.loc[a_player,'loses'] = 1

    elif hlose:

        pl_po.loc[a_player,'wins'] = 1

        pl_po.loc[h_player,'loses'] = 1

        

    if not atg :

        pl_po.loc[h_player,'csheet'] = 1



    if not htg :

        pl_po.loc[a_player,'csheet'] = 1





    if len(all_cards) <= n :

        print("Index error: Number of table rows: ", n)

    else :

        cards = all_cards[n]['value']

        card_g = card_status(cards)



    # Yellow and red card information

    yc = card_g[0]

    yh = card_g[1]

    y2c = card_g[2]

    y2h = card_g[3]

    rc = card_g[4]

    rh = card_g[5]

    

    for pid in yh :

        ci = pl_po[pl_po['id'] == pid].index

        pl_po.loc[ci,'ycards'] = pl_po.loc[ci,'ycards'] + 1



    for pid in y2h :

        ci = pl_po[pl_po['id'] == pid].index

        pl_po.loc[ci,'y2cards'] = pl_po.loc[ci,'y2cards'] + 1



    for pid in rh :

        ci = pl_po[pl_po['id'] == pid].index

        pl_po.loc[ci,'rcards'] = pl_po.loc[ci,'rcards'] + 1

        

    all_plpo = all_plpo.append(pl_po,ignore_index=True)

          

    n = n + 1

    

print("Added: ", len(all_plpo), i)

# Calculate all positions together

u_plpo = all_plpo.groupby('id').sum()

print(len(u_plpo))



# Calculate number of games/player

u_plpo.loc[:,'games'] = u_plpo['player_Y1'] + u_plpo['player_Y2'] + u_plpo['player_Y3']+ u_plpo['player_Y4']+ u_plpo['player_Y5']+ u_plpo['player_Y6']+ u_plpo['player_Y7']+ u_plpo['player_Y8']+ u_plpo['player_Y9']+ u_plpo['player_Y10']+ u_plpo['player_Y11']



# Group accordgin the player positions

if DEBUG :

    ngk = len(u_plpo[u_plpo['player_Y1']/u_plpo['games'] > 0.5])

    nde = len(u_plpo[(u_plpo['player_Y2'] + u_plpo['player_Y3'] + u_plpo['player_Y4'])/u_plpo['games'] >= 0.5])

    nmf = len(u_plpo[(u_plpo['player_Y5'] + u_plpo['player_Y6'] + u_plpo['player_Y7'] + u_plpo['player_Y8'])/u_plpo['games'] > 0.5])

    nfw = len(u_plpo[(u_plpo['player_Y9'] + u_plpo['player_Y10'] + u_plpo['player_Y11'])/u_plpo['games'] >= 0.5])

    print(ngk, nde, nmf, nfw, ngk+nde+nmf+nfw)



# Goalkeepers

ugk = u_plpo[u_plpo['player_Y1']/u_plpo['games'] > 0.5]

ugk.loc[:,'POS'] = 'GK'

all_p = pd.DataFrame(ugk['POS'])



# Defenders

ude = u_plpo[(u_plpo['player_Y3'])/u_plpo['games'] >= 0.5]

ude.loc[:,'POS'] = 'DE'

all_p = all_p.append(pd.DataFrame(ude['POS']))



# Midfielders

umf = u_plpo[(u_plpo['player_Y5'] + u_plpo['player_Y6'] + u_plpo['player_Y7'] + u_plpo['player_Y8'])/u_plpo['games'] > 0.5]

umf.loc[:,'POS'] = 'MF'

all_p = all_p.append(pd.DataFrame(umf['POS']))



# Forwarders

ufw = u_plpo[(u_plpo['player_Y9'] + u_plpo['player_Y10'] + u_plpo['player_Y11'])/u_plpo['games'] >= 0.5]

ufw.loc[:,'POS'] = 'FW'

all_p = all_p.append(pd.DataFrame(ufw['POS']))



# Remove duplicates

all_p = all_p.groupby('id').max()



# Merge all gouprs back again

u_plpo = u_plpo.merge(all_p, left_index=True, right_index=True, how='left')

print(len(u_plpo[u_plpo['POS']=='GK']),len(u_plpo[u_plpo['POS']=='DE']),len(u_plpo[u_plpo['POS']=='MF']),len(u_plpo[u_plpo['POS']=='FW']))

# Remove unnecessary Y-location columns

for p_pos in player_pos :

    if p_pos in u_plpo.columns :

        del u_plpo[p_pos]
# Score goalkeepers

all_gk = u_plpo[u_plpo['POS'] == 'GK']

all_gk.loc[:,'total'] = all_gk['games'] + all_gk['goals']*10 + all_gk['2goals'] + all_gk['3goals']*3 + all_gk['wgoal'] - all_gk['ogoals']*4 + all_gk['assists']*10 + all_gk['csheet']*5 + all_gk['wins'] - all_gk['loses'] - all_gk['ycards'] - all_gk['y2cards']*3 - all_gk['rcards']*5

# Score defenders

all_de = u_plpo[u_plpo['POS'] == 'DE']

all_de.loc[:,'total'] = all_de['games'] + all_de['goals']*6 + all_de['2goals'] + all_de['3goals']*3 + all_de['wgoal'] - all_de['ogoals']*4 + all_de['assists']*4 + all_de['csheet']*2 + all_de['wins'] - all_de['loses'] - all_de['ycards'] - all_de['y2cards']*3 - all_de['rcards']*5

# Score  midfielders

all_mf = u_plpo[u_plpo['POS'] == 'MF']

all_mf.loc[:,'total'] = all_mf['games'] + all_mf['goals']*4 + all_mf['2goals'] + all_mf['3goals']*3 + all_mf['wgoal'] - all_mf['ogoals']*4 + all_mf['assists']*3 + all_mf['wins'] - all_mf['loses'] - all_mf['ycards'] - all_mf['y2cards']*3 - all_mf['rcards']*5

# Score  midfielders

all_fw = u_plpo[u_plpo['POS'] == 'FW']

all_fw.loc[:,'total'] = all_fw['games'] + all_fw['goals']*3 + all_fw['2goals'] + all_fw['3goals']*3 + all_fw['wgoal'] - all_fw['ogoals']*4 + all_fw['assists']*3 + all_fw['wins'] - all_fw['loses'] - all_fw['ycards'] - all_fw['y2cards']*3 - all_fw['rcards']*5

#all_de.sort_values(by=['total'], ascending=False)
# Read from database and modify column

fifa_attr = pd.read_sql_query("SELECT player_api_id, overall_rating, date FROM Player_Attributes", cnx)

fifa_por = pd.DataFrame(fifa_attr)

fifa_por = fifa_por.rename(columns={"player_api_id": "id"})

#print("All players in DB: ",len(pl_a))



# Get additional Player information from database

fifa_player = pd.read_sql_query("SELECT player_api_id, player_name, birthday date FROM Player", cnx)

fifa_pname = pd.DataFrame(fifa_player)

fifa_pname = fifa_pname.rename(columns={"player_api_id": "id"})



# Use ratings only from 2 last year, use mean of multiple rating

fifa_por_lr = fifa_por[fifa_por['date'] > '2015-01-01']

del fifa_por_lr['date']

fifa_por_lr = fifa_por_lr.groupby('id', as_index=False).mean()



# Merge player informations 

fifa_por_lr = fifa_por_lr.merge(fifa_pname)



# Prepare League/Season score for comparison

ls_gk = pd.DataFrame(all_gk.index, columns=['id'])

ls_gk['fm_score'] = all_gk['total'].values



ls_de = pd.DataFrame(all_de.index, columns=['id'])

#ls_de = pd.DataFrame(all_de['id'], columns=['id'])

ls_de['fm_score'] = all_de['total'].values



ls_mf = pd.DataFrame(all_mf.index, columns=['id'])

ls_mf['fm_score'] = all_mf['total'].values



ls_fw = pd.DataFrame(all_fw.index, columns=['id'])

ls_fw['fm_score'] = all_fw['total'].values



# Merge with Goalkeeper data

combined_gk_r = ls_gk.merge(fifa_por_lr)

print("Merged GK: ",len(combined_gk_r))



# Merge with Defender data

combined_de_r = ls_de.merge(fifa_por_lr)

print("Merged DE: ",len(combined_de_r))



# Merge with Midfielder data

combined_mf_r = ls_mf.merge(fifa_por_lr)

print("Merged MF: ",len(combined_mf_r))



# Merge with Forward data

combined_fw_r = ls_fw.merge(fifa_por_lr)

print("Merged FW: ",len(combined_fw_r))

#combined_gk_r
sorted_gk_r = combined_gk_r.sort_values(by=['fm_score'], ascending=False)

sorted_gk_r['FM rank'] = combined_gk_r['fm_score'].rank(ascending=False)

sorted_fifa_r = sorted_gk_r.sort_values(by=['overall_rating'], ascending=False)

sorted_fifa_r['FIFA rank'] = sorted_fifa_r['overall_rating'].rank(ascending=False)

combined_gk_r = sorted_fifa_r.sort_values(by=['fm_score'], ascending=False)
combined_gk_r[:10]
f10_id = pd.Index(sorted_fifa_r['id'][:10])

fm10_id = pd.Index(combined_gk_r['id'][:10])

len(fm10_id & f10_id)
import matplotlib.pyplot as plt



fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('Goalkeepers - FM rank vs. Fifa rank',fontsize=10)

axis.set_xlabel('FM rank',fontsize=10)

axis.set_ylabel('FIFA rank',fontsize=10)



Xgk = combined_gk_r['FM rank']

Ygk = combined_gk_r['FIFA rank']



axis.scatter(Xgk, Ygk)

plt.show()
np.corrcoef(Xgk, Ygk)
sorted_de_r = combined_de_r.sort_values(by=['fm_score'], ascending=False)

sorted_de_r['FM rank'] = combined_de_r['fm_score'].rank(ascending=False)

sorted_fifa_r = sorted_de_r.sort_values(by=['overall_rating'], ascending=False)

sorted_fifa_r['FIFA rank'] = sorted_fifa_r['overall_rating'].rank(ascending=False)

combined_de_r = sorted_fifa_r.sort_values(by=['fm_score'], ascending=False)
combined_de_r[:10]
f10_id = pd.Index(sorted_fifa_r['id'][:10])

fm10_id = pd.Index(combined_de_r['id'][:10])

len(fm10_id & f10_id)
fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('Defenders - FM rank vs. Fifa rank',fontsize=10)

axis.set_xlabel('FM rank',fontsize=10)

axis.set_ylabel('FIFA rank',fontsize=10)



Xde = combined_de_r['FM rank']

Yde = combined_de_r['FIFA rank']



axis.scatter(Xde, Yde)

plt.show()
np.corrcoef(Xde, Yde)
sorted_mf_r = combined_mf_r.sort_values(by=['fm_score'], ascending=False)

sorted_mf_r['FM rank'] = combined_mf_r['fm_score'].rank(ascending=False)

sorted_fifa_r = sorted_mf_r.sort_values(by=['overall_rating'], ascending=False)

sorted_fifa_r['FIFA rank'] = sorted_fifa_r['overall_rating'].rank(ascending=False)

combined_mf_r = sorted_fifa_r.sort_values(by=['fm_score'], ascending=False)
combined_mf_r[:10]
f10_id = pd.Index(sorted_fifa_r['id'][:10])

fm10_id = pd.Index(combined_mf_r['id'][:10])

len(fm10_id & f10_id)
fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('Midfielders - FM rank vs. Fifa rank',fontsize=10)

axis.set_xlabel('FM rank',fontsize=10)

axis.set_ylabel('FIFA rank',fontsize=10)



Xmf = combined_mf_r['FM rank']

Ymf = combined_mf_r['FIFA rank']



axis.scatter(Xmf, Ymf)

plt.show()
np.corrcoef(Xmf, Ymf)
sorted_fw_r = combined_fw_r.sort_values(by=['fm_score'], ascending=False)

sorted_fw_r['FM rank'] = combined_fw_r['fm_score'].rank(ascending=False)

sorted_fifa_r = sorted_fw_r.sort_values(by=['overall_rating'], ascending=False)

sorted_fifa_r['FIFA rank'] = sorted_fifa_r['overall_rating'].rank(ascending=False)

combined_fw_r = sorted_fifa_r.sort_values(by=['fm_score'], ascending=False)
combined_fw_r[:10]
f10_id = pd.Index(sorted_fifa_r['id'][:10])

fm10_id = pd.Index(combined_fw_r['id'][:10])

len(fm10_id & f10_id)
fig, axis = plt.subplots()

# Grid lines, Xticks, Xlabel, Ylabel



axis.yaxis.grid(True)

axis.set_title('Forwarders - FM rank vs. Fifa rank',fontsize=10)

axis.set_xlabel('FM rank',fontsize=10)

axis.set_ylabel('FIFA rank',fontsize=10)



Xfw = combined_fw_r['FM rank']

Yfw = combined_fw_r['FIFA rank']



axis.scatter(Xfw, Yfw)

plt.show()
np.corrcoef(Xfw, Yfw)