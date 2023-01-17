

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
matches = pd.read_csv('../input/match.csv', index_col=0)

matches.head()
# Each file needs to be removed after use.

del matches

gc.collect()
players = pd.read_csv('../input/players.csv')

players.iloc[:5,:15]
players['account_id'].value_counts()
players.iloc[:5,20:30]
players.iloc[:5,40:55]
#cleanup

del players

gc.collect()
player_time = pd.read_csv('../input/player_time.csv')

player_time.head()
a_match = player_time.query('match_id == 1')
a_match.T
del player_time

gc.collect()
teamfights = pd.read_csv('../input/teamfights.csv')

teamfights.head()
del teamfights

gc.collect()
teamfights_players = pd.read_csv('../input/teamfights_players.csv')

teamfights_players.head()
del teamfights_players

gc.collect()
chat = pd.read_csv('../input/chat.csv')

chat.head()
# problem with the hero_ids in test_player brought to my attention by @Dexter, thanks!

# hero_id is 0 in 15 cases. 



test_players = pd.read_csv('../input/test_player.csv')

hero_names = pd.read_csv('../input/hero_names.csv')
# As can been seen the number of zeros appearing here are much less then the least popular hero. These are very likely

# caused by processing problems, either in my data generation code, or in the data pulled from steam. 

test_players['hero_id'].value_counts().tail()
test_players.query('hero_id == 0')
# remove matches with any invalid hero_ids

# imputing hero_id, is likely possible but the data is not available online in this dataset



matches_with_zero_ids = test_players.query('hero_id == 0')['match_id'].values.tolist()

test_players = test_players.query('match_id != @matches_with_zero_ids')
# check that the invalid ids are removed

# This is now on my list of bugs to fix for next release. 

test_players['hero_id'].value_counts().tail()
# player_ratings.csv contains trueskill ratings for players in the match, and test data.

# True Skill is a rating method somewhat like MMR, and can be used to sort players by skill. 



player_ratings = pd.read_csv('../input/player_ratings.csv')
player_ratings.head()
# Now create a list of player rankings by using the formula mu - 3*sigma

# This ranking formula penalizes players with fewer matches because there is more uncertainty



player_ratings['conservative_skill_estimate'] = player_ratings['trueskill_mu'] - 3*player_ratings['trueskill_sigma']
player_ratings.head()
player_ratings = player_ratings.sort_values(by='conservative_skill_estimate', ascending=False)
# negative account ids are players not appearing in other data available in this dataset.



player_ratings.head(10)
del player_ratings

gc.collect()
match_outcomes = pd.read_csv('../input/match_outcomes.csv')
# each match has data on two rows. the 'rad' tells whether the team is Radiant or not(1 is Radiant 0 is Dire)

# negative account ids are not in the other available data. account_id 0 is for anonymous players.

match_outcomes.head()
del match_outcomes

gc.collect()
ability_upgrades = pd.read_csv('../input/ability_upgrades.csv')

ability_ids = pd.read_csv('../input/ability_ids.csv')
ability_ids.head()
ability_upgrades.head()
del ability_upgrades, ability_ids

gc.collect()
purchase_log = pd.read_csv('../input/purchase_log.csv')

item_ids = pd.read_csv('../input/item_ids.csv')
item_ids.head()
purchase_log.head()
del purchase_log, item_ids

gc.collect()