# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Here I input the revelant data using Pandas.
vid_rev_DF = pd.read_csv('../input/video_review.csv')
player_data_DF = pd.read_csv('../input/play_player_role_data.csv')
play_info_DF = pd.read_csv('../input/play_information.csv')
player_punt_DF = pd.read_csv('../input/player_punt_data.csv')
# Quick test to see what the data looks like.
vid_rev_DF.tail()
# Here I begin to store all of the revelant information in arrays.
Concussed_position = []; # the position that a concussed player was playing on the punt play
C_YdLines = []; # the yard line the ball was snapped from on the punt play
C_Punting_Team = []; # 
traditional_C_pos = [];
player_number = [];
player_team = [];
GameIDs = np.array(vid_rev_DF["GameKey"].values);
PlayIDs = np.array(vid_rev_DF["PlayID"].values);
GSISIDs = np.array(vid_rev_DF["GSISID"].values);


Guard_C_plays=[];

for i in range(len(GameIDs)):
    Game = GameIDs[i];
    Play = PlayIDs[i];
    Player = GSISIDs[i];
    
    Concussed_player_trad_pos = player_punt_DF.loc[(player_punt_DF['GSISID'] == Player)];
    traditional_C_pos.append(Concussed_player_trad_pos.iloc[0]['Position']);
    player_number.append(Concussed_player_trad_pos.iloc[0]['Number']);
    
   # if "d" not in Concussed_player_trad_pos.iloc[0]['Number']:
   #     player_team.append
        
   # else: 
    
    Concussion_play = play_info_DF.loc[(play_info_DF['PlayID'] == Play)
                                         & (play_info_DF['GameKey'] == Game)];
    C_YdLines.append(Concussion_play.iloc[0]['YardLine']);
    C_Punting_Team.append(Concussion_play.iloc[0]['Poss_Team']);
    
    Concussed_player = player_data_DF.loc[(player_data_DF['PlayID'] == Play)
                                         & (player_data_DF['GameKey'] == Game)
                                        & (player_data_DF['GSISID'] == Player)];
    #print(Concussed_player);
    Concussed_position.append(Concussed_player.iloc[0]['Role']);
    
    if ((Concussed_player.iloc[0]['Role']=='PLG') or Concussed_player.iloc[0]['Role']=='PRG'):
        Guard_C_plays.append(Concussed_player.iloc[0]['PlayID'])
        #Guard_C_

String_counts1 = Counter(Concussed_position)

#frequencies = String_counts.values()
#names = String_counts.keys()

#x_coordinates = np.arange(len(String_counts))
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.bar(x_coordinates, frequencies, align='center')
#ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
#ax.xaxis.set_major_formatter(plt.FixedFormatter(names))
#plt.show()
df = pd.DataFrame.from_dict(String_counts1, orient='index')
plot = df.plot(kind='bar', figsize = (9, 6),
               title="Concussed Player Position on Punt Return", legend=False,
                fontsize = 18)
plot.title.set_size(24);
plot.set_xlabel("Player Position", fontsize=20);
plot.set_ylabel("Number of Concussions", fontsize= 20);
Punt_Positions = {
    # Punt Coverage Positions
    'PLG': 'Guard',
    'PRG': 'Guard',
    'PRT': 'Tackle',
    'PLT': 'Tackle',
    'GR': 'Gunner',
    'GL': 'Gunner',
    'PRW': 'Wing',
    'PLW': 'Wing',
    'PLS': 'Center',
    'P': 'Punter',
    'PPR': 'Protector',
    
    # Punt Return Positions
    'PR': 'Returner',
    'PDR1': 'DLine',
    'PDR2': 'DLine',
    'PDR3': 'DLine',
    'PDL1': 'DLine',
    'PDL2': 'DLine',
    'PDL3': 'DLine',
    'PLR': 'Linebacker',
    'PLM': 'Linebacker',
    'PLL': 'Linebacker',
    'VR': 'Cornerback',
    'PFB': 'Ret Protector'
}

Punt_Cov = 0;
Punt_Ret = 0;
Concussed_position2 = [Punt_Positions[k] for k in Concussed_position]

for i in range(len(Concussed_position2)):
    pos = Concussed_position2[i];
    if ((pos == 'Guard') or (pos == 'Tackle') or (pos == 'Gunner') or (pos == 'Wing') or (pos == 'Center')
       or (pos == 'Punter') or (pos == 'Protector')):
        Punt_Cov += 1;
    else: Punt_Ret += 1;
String_counts2 = Counter(Concussed_position2)

#frequencies = String_counts.values()
#names = String_counts.keys()

#x_coordinates = np.arange(len(String_counts))
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.bar(x_coordinates, frequencies, align='center')
#ax.xaxis.set_major_locator(plt.FixedLocator(x_coordinates))
#ax.xaxis.set_major_formatter(plt.FixedFormatter(names))
#plt.show()
df = pd.DataFrame.from_dict(String_counts2, orient='index')
plot = df.plot(kind='bar', figsize = (9, 6),
               title="Concussed Player Position on Punt Return", legend=False,
                fontsize = 12)
plot.title.set_size(24);
plot.set_xlabel("Player Position", fontsize=20);
plot.set_ylabel("Number of Concussions", fontsize= 20);
plt.pie([Punt_Cov, Punt_Ret], explode=(0.1, 0), labels=('Coverage Team', 'Return Team'),
        autopct='%1.1f%%', shadow=True, startangle=180);
plt.axis('equal');
plt.show()
plt.pie([12, 25], explode=(0.1, 0), labels=('Launch Plays', 'Other'),
        autopct='%1.1f%%', shadow=True, startangle=180);
plt.axis('equal');
plt.show()
C_plays = {'Game ID': [5, 29, 189, 231, 234, 281, 296, 357, 384, 506, 553, 618],
     'Play ID': [3129, 538, 3509, 1976, 3278, 1521, 2667, 3630, 183, 1988, 1683, 2792]}
C_plays_df = pd.DataFrame(data=C_plays)
C_plays_df
