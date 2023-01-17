# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import copy



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



#

# --- Read inputs ---

#



injury = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

play = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

#- track = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")



print("A Preliminary Study on Possible Injuries Caused by Synthetic Surfaces")

print()



#

# --- Number of injury cases between two surface types: Natural and Synthetic ---

#



injury_Natural = injury.loc[injury['Surface'] == 'Natural']

injury_Synthetic = injury.loc[injury['Surface'] == 'Synthetic']

n_Natural = len(injury_Natural)

n_Synthetic = len(injury_Synthetic)

del injury_Natural

del injury_Synthetic



print("ANALYSIS 1. Sample distribution of injury cases between the two surface types: Natual and Synthetic")

print("  + Number of cases of injuries on the two surface types are fairly close, which is good.")

print("  + These counts will be used to \"normalize\" results throughout the comparison analyses.")

print("------------------------------------------------")

print("Number of injury cases on Natural surface = 48")

print("Number of injury cases on Synthetic surface = 57")

print("------------------------------------------------")

print("")



injury_norm = copy.deepcopy(injury)

injury_norm['DM_M1'] = np.where(injury_norm['Surface'] == 'Natural', injury_norm['DM_M1']/float(n_Natural), injury_norm['DM_M1']/float(n_Synthetic))

injury_norm['DM_M7'] = np.where(injury_norm['Surface'] == 'Natural', injury_norm['DM_M7']/float(n_Natural), injury_norm['DM_M7']/float(n_Synthetic))

injury_norm['DM_M28'] = np.where(injury_norm['Surface'] == 'Natural', injury_norm['DM_M28']/float(n_Natural), injury_norm['DM_M28']/float(n_Synthetic))

injury_norm['DM_M42'] = np.where(injury_norm['Surface'] == 'Natural', injury_norm['DM_M42']/float(n_Natural), injury_norm['DM_M42']/float(n_Synthetic))



#

# --- Overall injuries by surface type ---

#



injury_bySurface = injury_norm.groupby(['Surface'])[['DM_M1','DM_M7','DM_M28','DM_M42']].sum()

injury_bySurface['DM_all'] = injury_bySurface[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)



print("ANALYSIS 2. Overall injuries between Natural and Synthetic surface type")

print("  + DM_all = DM_M1 + DM_M7 + DM_M28 + DM_M42, represents all injury types together.")

print("  + Normalization is done by dividing the raw sums by respective total injury cases.")

print("  + Overall, it is about even for injuries to happen on Natural and Synthetic surfaces.")

print("  + A noticeable difference is that dm_m28 type injury is more likely to happen with ")

print("  +   Synthetic surface (55.3% on Synthetic surface versus 44.7% on Natural surface).")

print("  + All together, the Synthetic surface is slightly more like to cause injuries,")

print("  +   50.8% vs 49.2% on Natural surface.")

print("--------------------------------------------------------")

injury_bySurface['DM_all'] = injury_bySurface[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

print(injury_bySurface)

print("--------------------------------------------------------")

print()



#

# --- Comparing injuries to different body parts between two surface types ---

#



print("ANALYSIS 3. Comparing injuries to different body parts between two surface types")

print("  + Ankle and Toe injuries are much more likely to happen on Synthetic surface.")

print("  + While dm_m28 injury is overall more likely on Synthetic surface, the knee-related dm_m28 injury")

print("    were mostly occurred on the Natural surface.")

print("--------------------------------------------------------")

injury_bodypart_bySurface = injury_norm.groupby(['BodyPart','Surface'])[['DM_M1','DM_M7','DM_M28','DM_M42']].sum()

injury_bodypart_bySurface['DM_all'] = injury_bodypart_bySurface[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

print(injury_bodypart_bySurface)

print("--------------------------------------------------------------------")

print("")



#

# Comparing injuries by rosterposition between the two surfaces

#



print("ANALYSIS 4. Comparing injuries by rosterposition between the two surface types")

print("  + Most likely injuries on synthetic surface: Cornerback and Wide Receiver.")

print("  + Least likely injuries on synthetic surface: Offeneive Lineman.")

print("  + Most type of injuries: Cornerback: dm_m1; Wide Receiver: dm_m7, dm_m28, dm_m42.")

print("-----------------------------------------------------------------------------")

injury_play=injury_norm.merge(play,on=['PlayerKey','GameID','PlayKey'])

injury_roster = injury_play.groupby(['RosterPosition','Surface'])[['DM_M1','DM_M7','DM_M28','DM_M42']].sum()

injury_roster['DM_all'] = injury_roster[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

print(injury_roster)

print("-----------------------------------------------------------------------------")

print()

del injury_roster



#

# Comparing injuries by playtype between the two surfaces

#



injury_play['PlayType'] = np.where(injury_play['PlayType']=='Kickoff Not Returned','Kickoff',injury_play['PlayType'])

injury_play['PlayType'] = np.where(injury_play['PlayType']=='Kickoff Returned','Kickoff',injury_play['PlayType'])

injury_play['PlayType'] = np.where(injury_play['PlayType']=='Punt Returned','Punt',injury_play['PlayType'])

injury_play['PlayType'] = np.where(injury_play['PlayType']=='Punt Not Returned','Punt',injury_play['PlayType'])

injury_playtype = injury_play.groupby(['PlayType','Surface'])[['DM_M1','DM_M7','DM_M28','DM_M42']].sum()

injury_playtype['DM_all'] = injury_playtype[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)



print("ANALYSIS 5. Comparing injuries by playtype between the two surface types")

print("  + Both returned and not returned kickoffs are consolidated to kickoff")

print("  + Both returned and not returned punts are consolidated to punt")

print("  + Kickoff is significantly more likely on the synthetic for injuries of all types")

print("-----------------------------------------------------------------------------")

print(injury_playtype)

print("-----------------------------------------------------------------------------")

print()

del injury_playtype



print("ANALYSIS 6. Drilldown analysis")

print("  + So far we can conclud that ankle and toe injuries are most likely to happen on the ")

print("    synthetic surface to wide receivers during kickoffs.")

print("  + Now we do some drilldown analyses on the position and type of plays during injuries.")

print("  + We start by restricting results to kickoff plays as they are more likely for injuries.")

print()



#

# 6a

#



print("ANALYSIS 6a. Find position played during the kickoff injuries")

print("  + Results show WR much more likely to be injured during kickoff on the synthetic surface.")

print("    WR are probably playing as returners during kickoffs.")

print("  + Noticed slightly higher injuries on the natural surface during kickoff at other positions, ")

print("    which will not be the scope of this analysis.")

injury_play_Kickoff = injury_play.loc[injury_play['PlayType'] == 'Kickoff']

injury_position = injury_play_Kickoff.groupby(['Position','Surface'])[['DM_M1','DM_M7','DM_M28','DM_M42']].sum()

injury_position['DM_all'] = injury_position[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

print("-----------------------------------------------------------------------------")

print(injury_position)

print("-----------------------------------------------------------------------------")

print()

del injury_position



#

# 6b

#



print("ANALYSIS 6b. Find play types related to the CB injuries")

print("  + CB is also a high risk position according to ANALYSIS 4.")

print("  + CB injuries most likely happen during pass plays on the synthetic surface")

print("  + Injuries also occur with punt and rush plays on the synthetic surface, but ")

print("    no data on the natural surface, which could also mean injuries are more ")

print("    likely with those plays on the synthetic surface.")

injury_play_CB = injury_play.loc[injury_play['Position'] == 'CB']

injury_playtype = injury_play_CB.groupby(['PlayType','Surface'])[['DM_M1','DM_M7','DM_M28','DM_M42']].sum()

injury_playtype['DM_all'] = injury_playtype[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

print("-----------------------------------------------------------------------------")

print(injury_playtype)

print("-----------------------------------------------------------------------------")

print()

del injury_playtype



#

# 6c

#



print("ANALYSIS 6c. Combined findings of WR and CB plays")

print("  + We have determined that most likely synthetic surface injurues occur with WR at kickoff ")

print("    CB at pass plays. We combine the results for these positions and play types.")

print("  + The combined results show the comparison values of injuries for kickoff and pass plays ")

print("    by CB and WR.")

injury_play_combine = injury_play.loc[((injury_play['Position'] == 'CB')|(injury_play['Position'] == 'WR')) & \

                            ((injury_play['PlayType'] == 'Kickoff')|(injury_play['PlayType'] == 'Pass'))]

injury_combine = injury_play_combine.groupby(['Position','PlayType','Surface'])[['DM_M1','DM_M7','DM_M28','DM_M42']].sum()

injury_combine['DM_all'] = injury_combine[['DM_M1','DM_M7','DM_M28','DM_M42']].sum(axis=1)

print("-----------------------------------------------------------------------------")

print(injury_combine)

print("-----------------------------------------------------------------------------")

print()



#

# 6d Weather a factor

#



print("ANALYSIS 6d. Weather a factor?")

print("  + We did not have conclusive finding about the weather factor for the analyses that ")

print("    we focused on, that is, wide receivers at kickoff plays and corner back at pass plays.")

print("  + The results below is an offline analysis on the weather and no striking evidence to ")

print("    show the weather factor (values are raw counts).")

print("-----------------------------------------------------------------------------")

print("Position  PlayType  Surface    Weather       DM_M1 DM_M7 DM_M28 DM_M42 DM_all")

print("CB        Pass      Natural    Rainy            1    1     0     0      2")

print("CB        Pass      Synthetic  Indoor           4    3     1     1      9")

print("WR        Kickoff   Natural    Partly Cloudy    1    1     0     0      2")

print("WR        Kickoff   Synthetic  Cloudy           2    2     2     2      8")

print("WR        Kickoff   Synthetic  Partly Cloudy    1    1     1     1      4")

print("WR        Pass      Natural    Clear            1    1     0     0      2")

print("WR        Pass      Natural    Cloudy           1    0     0     0      1")

print("WR        Pass      Natural    Partly Cloudy    1    0     0     0      1")

print("WR        Pass      Natural    Rainy            1    0     0     0      1")

print("WR        Pass      Synthetic  Clear            2    1     0     0      3")

print("WR        Pass      Synthetic  Indoor           1    1     0     0      2")

print("WR        Pass      Synthetic  Partly Cloudy    1    1     1     0      3")

print("-----------------------------------------------------------------------------")

print()

print("==========")

print("CONCLUSION")

print("==========")

print("""

This is a very basic and limited comparison analysis between Natural vs Synthetic surfaces based

on the injury cases provided. There is no attempt to study the appropriateness of the sampling.

A quick evaluation was made to determine the importance of factors such as weather and temperature,

but we did not reach any conclusive results for out analysis. These factors should certainly be

examined through case studies, which we did not do here. The results here are statistical and without

investigation into individual cases. In general, we conclude

(1) In general, injuries on the Synthetic surface is only slightly more likely in most cases.

(2) The most striking injury cases on the Synthetic surfaces are from wide receivers at kickoff plays

    and corner backs at pass plays.

(3) The most synthetic surface injuries occurred to toes and ankles.

(4) The most corner back injuries are DM_M1, and the wide receiver injuries can be all types with

    DM_M1 the least.

(5) The study did not find significant results from factors such as weather and temperature.

""")