# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
shots = pd.read_csv('/kaggle/input/data.csv')
shots1 = shots.drop(['game_id', 'game_event_id', 'lat', 'lon', 'shot_zone_range', 'shot_zone_basic', 'team_id', 'team_name', 'matchup'], axis = 1)

for i in shots1[['action_type', 'combined_shot_type', 'period', 'playoffs', 'season', 'shot_type', 'shot_zone_area', 'opponent']]:
   	#cat_columns.append(shots1.columns.get_loc(i))
	shots1.loc[:, i] = shots1.loc[:, i].astype('category')	
cat_columns = shots1.select_dtypes(['category']).columns
shots1[cat_columns] = shots1[cat_columns].apply(lambda x: x.cat.codes)

shots1.game_date = pd.to_datetime(shots1.game_date)

shots1 = shots1[['shot_made_flag', 'action_type', 'combined_shot_type', 'loc_x', 'loc_y', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_type', 'shot_zone_area', 'game_date', 'opponent', 'shot_id']]

train = shots1[np.isfinite(shots1['shot_made_flag'])]
# or train = shots1.dropna()
train['shot_made_flag'] = train['shot_made_flag'].astype('int')
train['shot_made_flag'] = train['shot_made_flag'].astype('category')

for i in train[['action_type', 'combined_shot_type', 'period', 'playoffs', 'season', 'shot_type', 'shot_zone_area', 'opponent']]:
   	#cat_columns.append(shots1.columns.get_loc(i))
	train.loc[:, i] = train.loc[:, i].astype('category')

# train = train.iloc[np.random.permutation(len(train))]

test = shots1[shots1.isnull().any(axis = 1)]
shots1_str = shots.drop(['game_id', 'game_event_id', 'lat', 'lon', 'shot_zone_range', 'shot_zone_basic', 'team_id', 'team_name', 'matchup'], axis = 1)
shots1_str.game_date = pd.to_datetime(shots1_str.game_date)
shots1_str = shots1_str[['shot_made_flag', 'action_type', 'combined_shot_type', 'loc_x', 'loc_y', 'minutes_remaining', 'period', 'playoffs', 'season', 'seconds_remaining', 'shot_distance', 'shot_type', 'shot_zone_area', 'game_date', 'opponent', 'shot_id']]
train_str = shots1_str[np.isfinite(shots1_str['shot_made_flag'])]
train_str.columns.tolist()
import matplotlib.pyplot as plt
X = train.combined_shot_type
Y = train.shot_made_flag
plt.bar(X, Y, facecolor='#9999ff', edgecolor='white')
# plt.tight_layout()
#plt.savefig('C:/Users/jiangy/Dropbox/learnings/Stats_data/Projects/Kobe/pairs.png')
plt.show()
train.combined_shot_type[:10]
train_str.combined_shot_type[:10]
