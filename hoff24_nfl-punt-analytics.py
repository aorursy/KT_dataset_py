#NFL Punt Analytics
#By Cory Hofmann
#Updated 2019-01-09

import pandas as pd
import numpy as np

#NFL provided data
game_data = pd.read_csv('game_data.csv')
all_punts = pd.read_csv('play_information.csv')
player_punt_position = pd.read_csv('play_player_role_data.csv') #Punt-specific position
player_position = pd.read_csv('player_punt_data.csv')
concussion_video = pd.read_csv('video_footage-injury.csv')
concussion_cause = pd.read_csv('video_review.csv')

#Add punt position for primary and partner to the concussion cause dataset
cause_w_position = pd.merge(concussion_cause, player_punt_position, on = ['Season_Year','GameKey','PlayID','GSISID'])

#primary partner contains unclear and NaN, needs to be converted to integer
cause_w_position.Primary_Partner_GSISID = cause_w_position.Primary_Partner_GSISID.replace(['Unclear',np.nan],0)
cause_w_position.Primary_Partner_GSISID = cause_w_position.Primary_Partner_GSISID.astype(np.int64)
cause_w_both_positions = pd.merge(cause_w_position, player_punt_position, left_on=['Primary_Partner_GSISID','PlayID','GameKey'], right_on=['GSISID','PlayID','GameKey'], how = 'left', suffixes = ('','_Secondary'))
del cause_w_both_positions['Season_Year_Secondary']
del cause_w_both_positions['GSISID_Secondary']

#Add play description and some game data
cause_w_play = pd.merge(cause_w_both_positions, concussion_video[['PlayDescription', 'gamekey', 'playid']], left_on=['GameKey', 'PlayID'], right_on=['gamekey','playid'])
del cause_w_play['gamekey']
del cause_w_play['playid']
cause_w_game = pd.merge(cause_w_play, game_data[['GameKey','Game_Day','Season_Type']], on='GameKey')

#What types of punting play result in a concussion?
cause_w_game.PlayDescription.str.count("fair catch").sum() #2 Fair Catches
cause_w_game.PlayDescription.str.count("out of bounds").sum() #0 Punts Out of Bounds
cause_w_game.PlayDescription.str.count("downed by").sum() #3 Downed by Punting team
cause_w_game.PlayDescription.str.count("Touchback").sum() #0 Touchbacks
#Almost all verified concussions were during return plays

#Categorize the concussed players, what positions are most common?
cause_w_game.Role.value_counts() #positions on punting team far more dangerous
cause_w_game.Role_Secondary.value_counts() #Most often results from contact with punt returner

#Categorize based on position, as per appendix
Punting_team_pos = ['P','GL','GR','PPR','PC','PLW','PLT','PLG','PLS','PRG','PRT','PRW']
Return_team_pos = ['PR','PFB','VRo','VLo','VRi','VR','PDR1','PDR2','PDR3','PDL3','PDL3','PDL2','PDL1','PLR','PLL','PLL1','PLM']

#Determine which team gets most concussions
cause_w_game['Role_Team'] = np.empty([37,1])

for i in range(0,len(cause_w_game.Role)):
   if cause_w_game.Role[i] in Punting_team_pos:
      cause_w_game.Role_Team[i] = 'Punting Team'
   elif cause_w_game.Role[i] in Return_team_pos:
      cause_w_game.Role_Team[i] = 'Return Team'
   else:
       cause_w_game.Role_Team[i] = np.nan

cause_w_game['Role_Team_Secondary'] = np.empty([37,1])

for i in range(0,len(cause_w_game.Role_Secondary)):
   if cause_w_game.Role_Secondary[i] in Punting_team_pos:
      cause_w_game['Role_Team_Secondary'][i] = 'Punting Team'
   elif cause_w_game.Role_Secondary[i] in Return_team_pos:
      cause_w_game['Role_Team_Secondary'][i] = 'Return Team'
   else:
       cause_w_game['Role_Team_Secondary'][i] = np.nan
       
cause_w_game.Role_Team.value_counts() #positions on punting team far more dangerous than return team (27 vs 10)
cause_w_game.Role_Team_Secondary.value_counts() #Closer to even split for the concussion partner (Return team 18, Punting Team 15)

#Descriptive statistics on punting
len(all_punts) #6681 Total Unique Punting Plays
all_punts.GameKey.nunique() #662 Unique Games
all_punts.PlayDescription.str.count("fair catch").sum() #1659 fair catches
all_punts.PlayDescription.str.count("out of bounds").sum() #669 punts out of bounds
all_punts.PlayDescription.str.count("PENALTY").sum() #1078 penalties
all_punts.PlayDescription.str.count("downed by").sum() #811 downed by punting team
all_punts.PlayDescription.str.count("Touchback").sum() #408 Touchbacks
all_punts.PlayDescription.str.count("TOUCHDOWN").sum() #54 Touchdowns, but how many returns vs. other?
all_punts.PlayDescription.str.count("BLOCKED").sum() #29 Plays involved a block

test1 = all_punts.PlayDescription.str.contains('BLOCKED')
test2 = all_punts.PlayDescription.str.contains('TOUCHDOWN')
sum(test1 & test2) 
#9 Touchdowns resulting from blocks, thus 45 were returned for TD (0.67%)

#Supplemental Data from footballdb.com, all punt returns and kickoff returns in regular season!
#This will be used to show average punt return distance
#This will also be used to demonstrate that kickoffs are occuring less frequently due to recent rule changes
punt_ret_2016 = pd.read_csv('punt_ret_2016.csv')
punt_ret_2017 = pd.read_csv('punt_ret_2017.csv')
kickoff_ret_2010 = pd.read_csv('kickoff_ret_2010.csv') #in 2011, kickoff moved forward 5 yards
kickoff_ret_2015 = pd.read_csv('kickoff_ret_2015.csv') #before 2015 rule changes made
kickoff_ret_2018 = pd.read_csv('kickoff_ret_2018.csv') #before 2018 rule changes made

#Add column describing return ratio for returners (Number returns vs. Fair Catches)
punt_ret_2016['return_percent'] = punt_ret_2016.Num / (punt_ret_2016.Num + punt_ret_2016.FC)
punt_ret_2017['return_percent'] = punt_ret_2017.Num / (punt_ret_2017.Num + punt_ret_2017.FC)

#Let's see some information regarding an 'average' punt return using supplemental data
#Rather than attempt to parse PlayDescription, lets use this dataset
sum(punt_ret_2016.Yds) / sum(punt_ret_2016.Num) #Avg punt return = 8.6 yds
sum(punt_ret_2017.Yds) / sum(punt_ret_2017.Num) #Avg punt return = 8.2 yds

#Kickoffs may now be safer, but they are happening far less frequently down from 4/game to less than 2/game
#Major rule changes occured before 2011, 2015, and 2018 seasons
sum(kickoff_ret_2010.Num) #2033 attempted returns in 512 reg season games
sum(kickoff_ret_2015.Num) #1080 attempted returns in 512 reg season games
sum(kickoff_ret_2018.Num) #970 attempted returns in 512 reg season games

#Conclusions:
#Concussions occur more often during punts with a return
#Punt team members are at high risk, as they sprint downfield preparing to tackle the punt returner
#Rule changes to kickoffs have resulted in fewer kickoff returns (and fewer onsides kicks, as per other references)
#Penalties occur at a high rate -> Additional (potentially complex) rules during punting may increase this!

