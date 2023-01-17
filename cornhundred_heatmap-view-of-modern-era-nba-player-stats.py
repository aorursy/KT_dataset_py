show_widget = False
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from clustergrammer2 import net as net

df = {}

if show_widget == False:

    print('\n-----------------------------------------------------')

    print('>>>                                               <<<')    

    print('>>> Please set show_widget to True to see widgets <<<')

    print('>>>                                               <<<')    

    print('-----------------------------------------------------\n')    

    delattr(net, 'widget_class') 
df['season-ini'] = pd.read_csv('../input/Seasons_Stats.csv', index_col=0)

df['season-ini'].shape
df['season-ini'].tail()
cols = df['season-ini'].columns.tolist()

keep_cols = cols[-26:]

print(len(keep_cols))

mat = df['season-ini'][keep_cols].get_values()

print(mat.shape)
ser_year = df['season-ini']['Year']

ser_player = df['season-ini']['Player']

ser_age = df['season-ini']['Age']

ser_team = df['season-ini']['Tm']

ser_pos = df['season-ini']['Pos']

rows = []

for index in range(ser_player.shape[0]):

    inst_year = ser_year.iloc[index]

    inst_player = ser_player.iloc[index]

    inst_age = ser_age.iloc[index]

    inst_team = ser_team.iloc[index]

    inst_pos = ser_pos.iloc[index]

    inst_name = str(inst_player) + '_' + str(inst_year) + '_' + str(inst_team) + '_' + str(inst_age)

    inst_row = (inst_name, 'Team: ' + str(inst_team), 'Position: ' + str(inst_pos), 'Player: ' + str(inst_player))

    rows.append(inst_row)

print(len(rows))

print(len(list(set(rows))))
df['season'] = pd.DataFrame(data=mat, index=rows, columns=keep_cols).transpose()

df['season'].shape
keep_cols = [x for x in df['season'].columns.tolist() if float(x[0].split('_')[1]) >= 2005]

df['modern-era'] = df['season'][keep_cols]

df['modern-era'].shape
net.load_df(df['modern-era'])

net.swap_nan_for_zero()

net.normalize(axis='row', norm_type='zscore')

df['modern-era-z'] = net.export_df()

net.clip(-5,5)

net.load_df(net.export_df())

net.widget()