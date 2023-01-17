# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline



import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dota_ability = pd.read_csv("/kaggle/input/dota-2-matches/ability_ids.csv")

ability_upgrades = pd.read_csv("/kaggle/input/dota-2-matches/ability_upgrades.csv")

chat = pd.read_csv("/kaggle/input/dota-2-matches/chat.csv")

cluster_regions = pd.read_csv("/kaggle/input/dota-2-matches/cluster_regions.csv")

hero_names = pd.read_csv("/kaggle/input/dota-2-matches/hero_names.csv")

item_ids = pd.read_csv("/kaggle/input/dota-2-matches/item_ids.csv")

match_outcomes = pd.read_csv("/kaggle/input/dota-2-matches/match_outcomes.csv")

match = pd.read_csv("/kaggle/input/dota-2-matches/match.csv")

objectives = pd.read_csv("/kaggle/input/dota-2-matches/objectives.csv")

patch_dates = pd.read_csv("/kaggle/input/dota-2-matches/patch_dates.csv")

player_ratings = pd.read_csv("/kaggle/input/dota-2-matches/player_ratings.csv")

player_time = pd.read_csv("/kaggle/input/dota-2-matches/player_time.csv")

players = pd.read_csv("/kaggle/input/dota-2-matches/players.csv")

purchase_log = pd.read_csv("/kaggle/input/dota-2-matches/purchase_log.csv")

teamfights_players = pd.read_csv("/kaggle/input/dota-2-matches/teamfights_players.csv")

teamfights = pd.read_csv("/kaggle/input/dota-2-matches/teamfights.csv")

test_labels = pd.read_csv("/kaggle/input/dota-2-matches/test_labels.csv")

test_player = pd.read_csv("/kaggle/input/dota-2-matches/test_player.csv")
#get correct date in match df

match.loc[:,'start_time'] = pd.to_datetime(match.loc[:,'start_time'], unit='s')

match.head(2)
#add name of hero in players df

b = hero_names.drop(columns=['name'])

b.head(2)
players_heroname = players.merge(b, on='hero_id') 
players_heroname = players_heroname[['match_id','localized_name', 'account_id', 'hero_id', 'player_slot', 'gold',

       'gold_spent', 'gold_per_min', 'xp_per_min', 'kills', 'deaths',

       'assists', 'denies', 'last_hits', 'stuns', 'hero_damage',

       'hero_healing', 'tower_damage', 'item_0', 'item_1', 'item_2', 'item_3',

       'item_4', 'item_5', 'level', 'leaver_status', 'xp_hero', 'xp_creep',

       'xp_roshan', 'xp_other', 'gold_other', 'gold_death', 'gold_buyback',

       'gold_abandon', 'gold_sell', 'gold_destroying_structure',

       'gold_killing_heros', 'gold_killing_creeps', 'gold_killing_roshan',

       'gold_killing_couriers', 'unit_order_none',

       'unit_order_move_to_position', 'unit_order_move_to_target',

       'unit_order_attack_move', 'unit_order_attack_target',

       'unit_order_cast_position', 'unit_order_cast_target',

       'unit_order_cast_target_tree', 'unit_order_cast_no_target',

       'unit_order_cast_toggle', 'unit_order_hold_position',

       'unit_order_train_ability', 'unit_order_drop_item',

       'unit_order_give_item', 'unit_order_pickup_item',

       'unit_order_pickup_rune', 'unit_order_purchase_item',

       'unit_order_sell_item', 'unit_order_disassemble_item',

       'unit_order_move_item', 'unit_order_cast_toggle_auto',

       'unit_order_stop', 'unit_order_taunt', 'unit_order_buyback',

       'unit_order_glyph', 'unit_order_eject_item_from_stash',

       'unit_order_cast_rune', 'unit_order_ping_ability',

       'unit_order_move_to_direction', 'unit_order_patrol',

       'unit_order_vector_target_position', 'unit_order_radar',

       'unit_order_set_item_combine_lock', 'unit_order_continue']]

players_heroname.head(2)
#add heroname to match df

b = hero_names.drop(columns=['name'])

a = test_player.merge(b, on='hero_id') 

a.head(2)
#add name of items to purchase_log and name of hero via player slot 

players_heroname_slot = players_heroname[['match_id','localized_name','hero_id','player_slot']]

new_df = pd.merge(players_heroname_slot, purchase_log,  how='left', left_on=['player_slot','match_id'], right_on = ['player_slot','match_id'])

new_df.head(2)
d = purchase_log.merge(item_ids, on='item_id') 

d.head(2)
#add name of ability

dota_ability_renamed = dota_ability.rename(columns={"ability_id": "ability"}, errors="raise")

e = ability_upgrades.merge(dota_ability_renamed, on='ability') 

e.head(2)
# get info about any match
#number_match_you_want_to_see

match_numb = 155
any_match_heroes = players_heroname.loc[lambda players_heroname: players_heroname['match_id'] == match_numb][['localized_name','player_slot']].sort_values(by='player_slot', ascending=True)

label_list = list(any_match_heroes['localized_name'])

any_match_heroes
any_match_player_time = player_time.loc[lambda player_time: player_time['match_id'] == match_numb]



any_match_player_time['times'] = any_match_player_time['times']/60

any_match_player_time.head(2)
any_match_player_time.plot(kind='line',x='times',y=['gold_t_0','gold_t_1','gold_t_2','gold_t_3','gold_t_4','gold_t_128','gold_t_129','gold_t_130','gold_t_131','gold_t_132'],figsize=(20,10),label=label_list,style='.-')



plt.show()
#make plot with match played per each day in df

graph_match_dates = match.groupby(match['start_time'].dt.date)['match_id'].count()
graph_match_dates.plot( figsize=(10,5),style='.-')

plt.show()
# graph with percentage of hero occurency from whole quantity of matches

player_stat = players_heroname.groupby(['localized_name'])['localized_name'].count().sort_values(ascending=False)

label_list_stat = list(player_stat.index)
(player_stat/500000*100).plot(figsize=(20,5),style='.-',x='s',rot=90)

plt.grid(True)

plt.xticks(list(range(0, 110)),label_list_stat)

plt.show()
#Find the most effective hero:

players_heroname_mean = players_heroname.groupby(['localized_name'])[['gold_per_min','xp_per_min','kills', 'deaths']].mean()
#gold per minute

players_heroname_mean['gold_per_min'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(players_heroname_mean['gold_per_min'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(300,players_heroname_mean['gold_per_min'].astype(int).max()+100,100)))

plt.show()
#xp per minute

players_heroname_mean['xp_per_min'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(players_heroname_mean['xp_per_min'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(300,players_heroname_mean['xp_per_min'].astype(int).max()+100,100)))

plt.show()
#average kills

players_heroname_mean['kills'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(players_heroname_mean['kills'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(7,players_heroname_mean['kills'].astype(int).max()+2)))

plt.show()
#average deaths

players_heroname_mean['deaths'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(players_heroname_mean['deaths'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(7,players_heroname_mean['kills'].astype(int).max()+2)))

plt.show()
#end of first part

# now lets try to do more dip analysis
#query only rating matches

rating_match = match[match['game_mode'] == 22]
#add to each hero was on succesfull at this match or not 

match_win = rating_match[['match_id','radiant_win']]

players_heroname_win = players_heroname.merge(match_win, on='match_id') 

players_heroname_win = players_heroname_win[['match_id','radiant_win','localized_name','account_id', 'hero_id', 'player_slot', 'gold',

       'gold_spent', 'gold_per_min', 'xp_per_min', 'kills', 'deaths',

       'assists', 'denies', 'last_hits', 'stuns', 'hero_damage',

       'hero_healing', 'tower_damage', 'item_0', 'item_1', 'item_2', 'item_3',

       'item_4', 'item_5', 'level', 'leaver_status', 'xp_hero', 'xp_creep',

       'xp_roshan', 'xp_other', 'gold_other', 'gold_death', 'gold_buyback',

       'gold_abandon', 'gold_sell', 'gold_destroying_structure',

       'gold_killing_heros', 'gold_killing_creeps', 'gold_killing_roshan',

       'gold_killing_couriers', 'unit_order_none',

       'unit_order_move_to_position', 'unit_order_move_to_target',

       'unit_order_attack_move', 'unit_order_attack_target',

       'unit_order_cast_position', 'unit_order_cast_target',

       'unit_order_cast_target_tree', 'unit_order_cast_no_target',

       'unit_order_cast_toggle', 'unit_order_hold_position',

       'unit_order_train_ability', 'unit_order_drop_item',

       'unit_order_give_item', 'unit_order_pickup_item',

       'unit_order_pickup_rune', 'unit_order_purchase_item',

       'unit_order_sell_item', 'unit_order_disassemble_item',

       'unit_order_move_item', 'unit_order_cast_toggle_auto',

       'unit_order_stop', 'unit_order_taunt', 'unit_order_buyback',

       'unit_order_glyph', 'unit_order_eject_item_from_stash',

       'unit_order_cast_rune', 'unit_order_ping_ability',

       'unit_order_move_to_direction', 'unit_order_patrol',

       'unit_order_vector_target_position', 'unit_order_radar',

       'unit_order_set_item_combine_lock', 'unit_order_continue']]



def f(row):

    if row['radiant_win'] == True and row['player_slot'] in [0,1,2,3,4]:

        val = 1

    elif row['radiant_win'] == False and row['player_slot'] in [128,129,130,131,132]:

        val = 1

    else:

        val = 0

    return val

players_heroname_win['player_wins'] = players_heroname_win.apply(f, axis=1)
#make data frame with only succesfull players

df_onlywins_players=players_heroname_win[players_heroname_win['player_wins'] == 1]
#make average statistic about each hero at match

df_onlywins_players_mean = df_onlywins_players.groupby(['localized_name'])[['gold_per_min','xp_per_min','kills', 'deaths']].mean()
#make graph with most pickable heroes from succesfull teams

player_stat_onlywins = df_onlywins_players.groupby(['localized_name'])['localized_name'].count().sort_values(ascending=False)

(player_stat_onlywins/243344*100).plot(figsize=(20,5),style='.-',x='s',rot=90)

plt.grid(True)

plt.xticks(list(range(0, 110)),label_list_stat)

plt.show()
#gold per minute in succesfull team

df_onlywins_players_mean['gold_per_min'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(df_onlywins_players_mean['gold_per_min'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(300,df_onlywins_players_mean['gold_per_min'].astype(int).max()+100,100)))

plt.show()
#xp per minute in succesfull team

df_onlywins_players_mean['xp_per_min'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(df_onlywins_players_mean['xp_per_min'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(300,df_onlywins_players_mean['xp_per_min'].astype(int).max()+100,100)))

plt.show()
#average kills in succesfull team

df_onlywins_players_mean['kills'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(df_onlywins_players_mean['kills'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(7,df_onlywins_players_mean['kills'].astype(int).max()+2)))

plt.show()
#average deaths in succesfull team

df_onlywins_players_mean['deaths'].sort_values(ascending=False).head(10).plot(figsize=(20,5),style='.-',x='s',rot=0)

plt.grid(True)

plt.xticks(list(range(0, 10)),list(df_onlywins_players_mean['deaths'].sort_values(ascending=False).head(10).index))

plt.yticks(list(range(7,df_onlywins_players_mean['deaths'].astype(int).max()+2)))

plt.show()