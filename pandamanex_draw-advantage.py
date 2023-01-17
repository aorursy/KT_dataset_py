# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



race = pd.read_csv("../input/races.csv")



runs = pd.read_csv("../input/runs.csv")

grp_draw = runs[['result' , 'draw']].groupby('draw').mean()



grp_draw.plot.bar()



# Any results you write to the current directory are saved as output.
draw15 = runs.loc[runs['draw'] == 15]

print (draw15.shape)



#It looks like there is only one record of draw==15



grp_draw = grp_draw[0:14]

grp_draw.plot.bar()
grp_draw -= grp_draw.min()

grp_draw.plot.bar()



#The mean result is disproportionate to the draw number
grp_draw['precentage'] = grp_draw['result']*100.0 / 14.0

# Assuming there are 14 horses on a run



print (grp_draw['precentage'])
race.head()
config_A = race.loc[race['config']=='A']

print(config_A.shape)
config_A.head(10)
config_A_raceid = config_A['race_id'].values
config_A_runs = runs.loc[runs['race_id'].isin(config_A_raceid)]
grp_draw_config_A = config_A_runs[['result' , 'draw']].groupby('draw').mean()

grp_draw_config_A.plot.bar()
print(runs.shape)

print(config_A_runs.shape)
grp_draw_config_A -= grp_draw_config_A.min()

grp_draw_config_A['precentage'] = grp_draw_config_A['result']*100.0 / 14.0

grp_draw_config_A['control'] = grp_draw['precentage']

print (grp_draw_config_A[['control' , 'precentage']])
ST_races = race.loc[race['venue']=='ST']

HV_races = race.loc[race['venue']=='HV']

print(ST_races.shape)

print(HV_races.shape)
u_config = race['config'].unique()

print (u_config)





#ST_turf_races = ST_races.loc[ST_races['config'] != 'AWT']

#ST_awt_races = ST_races.loc[ST_races['config'] == 'AWT']

#print (ST_turf_races.shape)

#print (ST_awt_races.shape)
ST_races_id = ST_races['race_id'].values

ST_runs = runs.loc[runs['race_id'].isin(ST_races_id)]

grp_draw_ST = ST_runs[['result' , 'draw']].groupby('draw').mean()

grp_draw_ST = grp_draw_ST[0:14]

grp_draw_ST.plot.bar()



HV_races_id = HV_races['race_id'].values

HV_runs = runs.loc[runs['race_id'].isin(HV_races_id)]

grp_draw_HV = HV_runs[['result' , 'draw']].groupby('draw').mean()

grp_draw_HV = grp_draw_HV[0:14]

grp_draw_HV.plot.bar()
HV_samples_each_draw = []

for i in range(1,15):

    HV_samples_each_draw.append(HV_runs.loc[HV_runs['draw'] == i].shape[0])

print (HV_samples_each_draw)



# It seems draws 13 & 14 are rarely used in HV
ST_samples_each_draw = []

for i in range(1,15):

    ST_samples_each_draw.append(ST_runs.loc[ST_runs['draw'] == i].shape[0])

print (ST_samples_each_draw)
grp_draw_ST -= grp_draw_ST.min()

grp_draw_ST['precentage'] = grp_draw_ST['result'] * 100.0 / 14.0



grp_draw_HV -= grp_draw_HV.iloc[0]['result']

grp_draw_HV['precentage'] = grp_draw_HV['result'] * 100.0 / 12.0



grp_draw_HV.iloc[12]['precentage'] = 'nan'

grp_draw_HV.iloc[13]['precentage'] = 'nan'



grp_draw['ST_precentage'] = grp_draw_ST['precentage']

grp_draw['HV_precentage'] = grp_draw_HV['precentage']





print (grp_draw[['precentage' , 'ST_precentage' , 'HV_precentage']])


