# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/shot_logs.csv')

#print (df['LOCATION'])

#print (df.columns)

#print (df.head())

df['SHOT_RESULT'] = df['SHOT_RESULT'].map({'made': 1, 'missed': 0})

df['W'] = df['W'].map({'W': 1, 'L': 0})

x =  df.groupby(['player_id','GAME_ID']).mean()

y =  df.groupby(['CLOSEST_DEFENDER_PLAYER_ID']).mean()

fgm_offense =  pd.DataFrame(pd.DataFrame(x['FGM']).reset_index().groupby('player_id').mean()['FGM'])

fgm_defense =  pd.DataFrame(y['FGM'])

fgm_offense.index.names = ['player_id']

fgm_defense.index.names = ['player_id']

fgm_offense.columns = ['FGM_O']

fgm_defense.columns = ['FGM_D']

merged = fgm_offense.join(fgm_defense , how = 'inner')

merged['Efficiency_rate'] = merged['FGM_O'] - merged['FGM_D']

eff_df = merged.sort('Efficiency_rate' , ascending = [0])



df_names = df[['player_id','player_name']].drop_duplicates()

df_names.set_index('player_id',inplace = True)





print (df_names.join(eff_df , how = 'inner')[['player_name' , 'Efficiency_rate']])

eff_joined = df_names.join(eff_df , how = 'inner')[['player_name' , 'Efficiency_rate']]



df = df.set_index('player_id')



df_joined =  df.join(eff_joined, lsuffix='_left', rsuffix='_right')



for_reggression = df_joined[['Efficiency_rate','W','GAME_ID']].drop_duplicates()[['Efficiency_rate','W','GAME_ID']]

from pandas.stats.api import ols

res = ols(y=for_reggression['W'], x=for_reggression[['Efficiency_rate']])

print (res)

# Any results y