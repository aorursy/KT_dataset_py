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
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
play_player_role_data = pd.read_csv("../input/play_player_role_data.csv")
game_data = pd.read_csv('../input/game_data.csv')
play_info = pd.read_csv('../input/play_information.csv')
player_punt_data = pd.read_csv('../input/player_punt_data.csv')
video_review = pd.read_csv('../input/video_review.csv')
ngs_pre_2016 = pd.read_csv('../input/NGS-2016-pre.csv')
ngs_reg_2016_1 = pd.read_csv('../input/NGS-2016-reg-wk1-6.csv')
ngs_reg_2016_2 = pd.read_csv('../input/NGS-2016-reg-wk7-12.csv')
ngs_reg_2016_3 = pd.read_csv('../input/NGS-2016-reg-wk13-17.csv')
ngs_post_2016 = pd.read_csv('../input/NGS-2016-post.csv')
ngs_pre_2017 = pd.read_csv('../input/NGS-2017-pre.csv')
ngs_reg_2017_1 = pd.read_csv('../input/NGS-2017-reg-wk1-6.csv')
ngs_reg_2017_2 = pd.read_csv('../input/NGS-2017-reg-wk7-12.csv')
ngs_reg_2017_3 = pd.read_csv('../input/NGS-2017-reg-wk13-17.csv')
ngs_post_2017 = pd.read_csv('../input/NGS-2017-post.csv')
len(play_info)
len(video_review)
ax = video_review['Player_Activity_Derived'].value_counts().plot(kind='bar', title ="Activity", figsize=(15, 10), fontsize=12)
ax.set_xlabel("Activity", fontsize=12)
plt.show()
ax = video_review['Primary_Impact_Type'].value_counts().plot(kind='bar', title ="Impact Type", figsize=(15, 10), fontsize=12)
ax.set_xlabel("Type of Collision", fontsize=12)
plt.show()
play_player_role_data['identifier'] = play_player_role_data['GameKey'].astype(str) + "_" + play_player_role_data['PlayID'].astype(str)
video_review['identifier'] = video_review['GameKey'].astype(str) + "_" + video_review['PlayID'].astype(str)

play_info['identifier'] = play_info['GameKey'].astype(str) + "_" + play_info['PlayID'].astype(str)

play_injuries = video_review.merge(play_info, left_on='identifier', right_on='identifier', how='inner')

fc = 'fair catch'
fair_catch = play_injuries.PlayDescription.str.count(fc).sum()
muff = 'MUFFS'
muffed = play_injuries.PlayDescription.str.count(muff).sum()
tb = 'Touchback'
touchback = play_injuries.PlayDescription.str.count(tb).sum()
down = 'downed'
downed = play_injuries.PlayDescription.str.count(down).sum()
ob = 'out of bounds'
out_of_bounds = play_injuries.PlayDescription.str.count(ob).sum()

returned = len(play_injuries) - fair_catch - muffed - touchback - downed - out_of_bounds
play_result_conc = pd.DataFrame({'touchback': [touchback], 'Fair Catch': [fair_catch], 'muffed': [muffed], 'out_of_bounds': [out_of_bounds], 'downed': [downed], 'returned': [returned]})
ax = play_result_conc.plot(kind='bar', title ="Result o Play", figsize=(15, 10), fontsize=12)
ax.set_xlabel("Result", fontsize=12)
plt.show()
play_player_role_data['identifier'] = play_player_role_data['GameKey'].astype(str) + "_" + play_player_role_data['PlayID'].astype(str)
player_injured = play_injuries.merge(play_player_role_data, how='inner', on=['identifier', 'GSISID'])

ax = player_injured.Role.value_counts().plot(kind='bar', title ="Role of Player", figsize=(15, 10), fontsize=12)
ax.set_xlabel("Role", fontsize=12)
plt.show()
return_team = player_injured.Role.isin(['PDL1','PDL2','PDL3','PDL4','PDL5','PDL6','PDM','PDR1','PDR2','PDR3','PDR4','PDR5','PDR6'
                ,'PFB','PLL','PLL1','PLL2','PLL3','PLM','PLM1','PLR','PLR1','PLR2','PLR3','PR','VL','VLi'
                ,'VLo','VR','VRi','VRo']).sum()
punt_team = player_injured.Role.isin(['GL','GLi','GLo','GR','GRi','GRo','P','PC','PLG','PLS','PLT','PLW','PPL','PPLi','PPLo'
                 ,'PPR','PPRi','PPRo','PRG','PRT','PRW']).sum()
teams = pd.DataFrame({'Returning Team': [return_team], 'Punting Team': [punt_team]})
ax = teams.plot(kind='bar', title ="Side o Ball w Conc", figsize=(15, 10), fontsize=12)
ax.set_xlabel("Team", fontsize=12)
plt.show()