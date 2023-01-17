import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
WNCAA_df = pd.read_csv('../input/wncaa-data/WNCAA_data.csv')

Increased_WNCAA_df = pd.read_csv('../input/wncaa-data/WNCAA_data.csv')

Decreased_WNCAA_df = pd.read_csv('../input/wncaa-data/WNCAA_data.csv')
WNCAA_df.head()
WNCAA_df.columns
WNCAA_df.shape
WNCAA_df['WPoints_From_2'] = WNCAA_df['WFGM2']*2

WNCAA_df['LPoints_From_2'] = WNCAA_df['LFGM2']*2

WNCAA_df['WPoints_From_3'] = WNCAA_df['WFGM3']*3

WNCAA_df['LPoints_From_3'] = WNCAA_df['LFGM3']*3
WNCAA_df.head()
Decreased_WNCAA_df['WPoints_From_2'] = WNCAA_df['WFGM2']*1

Decreased_WNCAA_df['LPoints_From_2'] = WNCAA_df['LFGM2']*1

Decreased_WNCAA_df['WPoints_From_3'] = WNCAA_df['WFGM3']*2

Decreased_WNCAA_df['LPoints_From_3'] = WNCAA_df['LFGM3']*2



Decreased_WNCAA_df['New_WScore'] = Decreased_WNCAA_df['WPoints_From_2'] + Decreased_WNCAA_df['WPoints_From_3'] + Decreased_WNCAA_df['WFTM'] 

Decreased_WNCAA_df['New_LScore'] = Decreased_WNCAA_df['LPoints_From_2'] + Decreased_WNCAA_df['LPoints_From_3'] + Decreased_WNCAA_df['LFTM'] 



Decreased_WNCAA_df['score_dif'] = Decreased_WNCAA_df['New_WScore'] - Decreased_WNCAA_df['New_LScore']



Decreased_WNCAA_df.head()
Increased_WNCAA_df['WPoints_From_2'] = WNCAA_df['WFGM2']*3

Increased_WNCAA_df['LPoints_From_2'] = WNCAA_df['LFGM2']*3

Increased_WNCAA_df['WPoints_From_3'] = WNCAA_df['WFGM3']*4

Increased_WNCAA_df['LPoints_From_3'] = WNCAA_df['LFGM3']*4



Increased_WNCAA_df['New_WScore'] = Increased_WNCAA_df['WPoints_From_2'] + Increased_WNCAA_df['WPoints_From_3'] + Increased_WNCAA_df['WFTM'] 

Increased_WNCAA_df['New_LScore'] = Increased_WNCAA_df['LPoints_From_2'] + Increased_WNCAA_df['LPoints_From_3'] + Increased_WNCAA_df['LFTM'] 



Increased_WNCAA_df['score_dif'] = Increased_WNCAA_df['New_WScore'] - Increased_WNCAA_df['New_LScore']



Increased_WNCAA_df.head()
Increased_Winner_Changes = Increased_WNCAA_df[Increased_WNCAA_df.score_dif <= 0].groupby(['Season']).size()

Decreased_Winner_Changes = Decreased_WNCAA_df[Decreased_WNCAA_df.score_dif <= 0].groupby('Season').size()
ax = Increased_Winner_Changes.plot.bar(y = 'Games Where Winner Changed')
ax2 = Decreased_Winner_Changes.plot.bar()