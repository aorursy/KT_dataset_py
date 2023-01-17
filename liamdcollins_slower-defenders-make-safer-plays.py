import os
import pandas as pd
import datetime
import ciso8601
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import math

seasons = ["pre", "reg-wk1-6", "reg-wk7-12", "reg-wk13-17", "post"]
years = ["2016", "2017"]

playerSpeeds = pd.Series()

for year in years:
    for season in seasons:
        currentFile = '../input/NGS-' + year + '-' + season + '.csv'
        print("Loading:", currentFile)
        ngsDataRaw = pd.read_csv(currentFile, parse_dates=['Time'])
        print("Processing...")
        # drop columns with no GSISID
        ngsData = ngsDataRaw.sort_values(['GameKey', 'PlayID', 'GSISID', 'Time'])
        ngsData['timeDiff'] = ngsData['Time'].diff()
        mask = ((ngsData.GSISID != ngsData.GSISID.shift(1)) | (ngsData.PlayID != ngsData.PlayID.shift(1)) | (ngsData.GameKey != ngsData.GameKey.shift(1)))
        ngsData['timeDiff'][mask] = np.nan
        ngsData['speed'] = ngsData['dis']/((ngsData['timeDiff'].dt.microseconds/(10**6)) + (ngsData['timeDiff'].dt.seconds))

        averagePlayerSpeeds = ngsData.groupby(['GSISID'], as_index=False)['speed'].mean()
        playerSpeeds = playerSpeeds.append(averagePlayerSpeeds)
   
averagePlayerSpeeds = averagePlayerSpeeds.groupby(['GSISID'])['speed'].mean()

playerSpeeds.head()

playerSpeeds['GSISID'] = playerSpeeds['GSISID'].astype(int)
ppr = pd.read_csv("../input/play_player_role_data.csv")
#ppr = ppr.join(other=playerSpeeds, on="GSISID")
ppr = pd.merge(ppr, playerSpeeds, on='GSISID', how='left')

a = ppr['Role'].unique()

pr = ['PDL2', 'PDR3', 'PLR2', 'PLR', 'PDR4', 'VRi', 'VRo',
       'VLo', 'PDL3', 'PLL', 'PLL2', 'PDR2', 'PDL5', 'PLM',
       'PR', 'PDL4', 'VL', 'PDL1', 'PDR1',
       'VLi', 'PLR1', 'PPLi', 'VR', 'PLL1',
       'PFB', 'PDR5', 'PDM', 'PDL6', 'PLL3', 'PLR3',
       'PDR6', 'PPLo', 'PLM1']

pc = list(set(a) - set(pr))

ppr_defense = ppr[ppr['Role'].isin(pr)]
ppr_offense = ppr[ppr["Role"].isin(pc)]

ppr_defense.head()
ppr_offense.head()


ppr_defense_per_play = ppr_defense.groupby(['GSISID'], as_index=False)['speed'].mean()
injuryPlays = pd.read_csv("../input/video_footage-injury.csv")
safePlays = pd.read_csv("../input/video_footage-control.csv")
safePlays.head()
ppr_defense["uniqueplay"] = ppr_defense["GameKey"].map(str) + ppr_defense["PlayID"].map(str)
ppr_offense["uniqueplay"] = ppr_offense["GameKey"].map(str) + ppr_offense["PlayID"].map(str)
safePlays["uniqueplay"] = safePlays["gamekey"].map(str) + safePlays["playid"].map(str)
injuryPlays["uniqueplay"] = injuryPlays["gamekey"].map(str) + injuryPlays["playid"].map(str)

speedWithDetails_defense_safe = pd.merge(ppr_defense, safePlays, on="uniqueplay", how='right')
speedWithDetails_defense_injury = pd.merge(ppr_defense, injuryPlays, on="uniqueplay", how='right')
speedWithDetails_offense_safe = pd.merge(ppr_offense, safePlays, on="uniqueplay", how='right')
speedWithDetails_offense_injury = pd.merge(ppr_offense, injuryPlays, on="uniqueplay", how='right')
meanSpeeds_safe = speedWithDetails_offense_safe.groupby(["uniqueplay"])['speed'].mean()
meanSpeeds_injury = speedWithDetails_offense_injury.groupby(["uniqueplay"])['speed'].mean()
print("Average speed of a player on kicking team on a play that results in injury:", meanSpeeds_injury.mean())
print("Average speed of a player on kicking team on a play that results in no injury:", meanSpeeds_safe.mean())
meanSpeeds_safe = speedWithDetails_defense_safe.groupby(["uniqueplay"])['speed'].mean()
meanSpeeds_injury = speedWithDetails_defense_injury.groupby(["uniqueplay"])['speed'].mean()
print("Average speed of a player on receiving team on a play that results in injury:", meanSpeeds_injury.mean())
print("Average speed of a player on receiving team on a play that results in no injury:", meanSpeeds_safe.mean())

