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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
df = pd.read_csv('../input/ronaldo/data.csv')
df.head(20)

df.describe()
df.apply(lambda x: sum(x.isnull()))
df['shot_id_number'].replace(np.nan, df['Unnamed: 0'] + 1, inplace = True)
submission = pd.read_csv('../input/sample-submission/sample_submission.csv')
test_df = df[df['shot_id_number'].isin(submission['shot_id_number'])]
train_df = df[~df['shot_id_number'].isin(submission['shot_id_number'])]
df['type_of_shot'] = df['type_of_shot'].fillna(df['type_of_combined_shot'])
df['location_x'].nunique()
plt.scatter(df['location_x'], df['location_y'])
df['location_x'] = df['location_x'].fillna(df['location_x'].mean())
avg_location_y = df.pivot_table(values = 'location_y', index = 'location_x')

miss_bool = df['location_y'].isnull()
df.loc[miss_bool, 'location_y'] = df.loc[miss_bool, 'location_x'].apply(lambda x: avg_location_y.at[x, 'location_y'])
df['remaining_min'].value_counts()
df['remaining_min'].hist()
df['remaining_min'] = df['remaining_min'].fillna(df['remaining_min'].median())
df['power_of_shot'].value_counts()
avg_power_of_shot = df.pivot_table(values = 'power_of_shot', index = 'type_of_shot')

miss_bool = df['power_of_shot'].isnull()

df.loc[miss_bool, 'power_of_shot'] = df.loc[miss_bool, 'type_of_shot'].apply(lambda x: avg_power_of_shot.at[x, 'power_of_shot'])
df['power_of_shot'] = df['power_of_shot'].astype(int)
df['knockout_match'].value_counts()
df['knockout_match'] = df['knockout_match'].fillna(0)
df['game_season_shift'] = df.game_season.shift(periods = 1)
df['game_season'] = df['game_season'].fillna(df['game_season_shift'])
df['game_season_shift'] = df.game_season.shift(periods = 2)
df['game_season'] = df['game_season'].fillna(df['game_season_shift'])
df['game_season_shift'] = df.game_season.shift(periods = 3)
df['game_season'] = df['game_season'].fillna(df['game_season_shift'])
df['remaining_sec'] = df['remaining_sec'].fillna(df['remaining_sec'].median())
avg_distance_of_shot = df.pivot_table(values = 'distance_of_shot', index = 'type_of_shot')

miss_bool = df['distance_of_shot'].isnull()

df.loc[miss_bool, 'distance_of_shot'] = df.loc[miss_bool, 'type_of_shot'].apply(lambda x: avg_distance_of_shot.at[x, 'distance_of_shot'])
df['distance_of_shot'] = df['distance_of_shot'].replace(31.981034, 32).astype(int)
df['area_of_shot'].value_counts()
df['area_of_shot'] = df['area_of_shot'].fillna('Center(C)')
df['home/away'].value_counts()
df['shot_basics'] = df['shot_basics'].fillna('Mid Range')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['shot_basics'] = le.fit_transform(df['shot_basics'])

df['range_of_shot'] = df['range_of_shot'].astype(str)

le_1 = LabelEncoder()

df['range_of_shot'] = le.fit_transform(df['range_of_shot'])

avg_range_of_shot = df.pivot_table(values = 'range_of_shot', index = ['shot_basics', 'distance_of_shot'])

miss_bool = df['range_of_shot'].isnull()

df.loc[miss_bool, 'range_of_shot'] = df.loc[miss_bool, ('shot_basics', 'distance_of_shot')].apply(lambda x: avg_range_of_shot.at[x, 'range_of_shot'])
df['team_name'] = df['team_name'].fillna('Manchester United')
df.columns
df = df.drop(['Unnamed: 0', 'match_event_id', 'team_name', 'date_of_game',

       'home/away', 'lat/lng', 'type_of_combined_shot', 'match_id', 'team_id', 'remaining_min.1',

       'power_of_shot.1', 'knockout_match.1', 'remaining_sec.1',

       'distance_of_shot.1'], axis = 1)
train_df['is_goal'].value_counts()
test_df = df[df['shot_id_number'].isin(submission['shot_id_number'])]
train_df = df[~df['shot_id_number'].isin(submission['shot_id_number'])]
train_df.apply(lambda x: sum(x.isnull()))
train_df.loc[:,'is_goal'] = train_df['is_goal'].fillna(0)
train_y = train_df.iloc[: , 8].values
train_X = train_df.iloc[:,:]
id_col = test_df['shot_id_number']
test_df = test_df.drop(['shot_id_number'], axis = 1)
test_X = test_df.iloc[:, :]
from sklearn.naive_bayes import GaussianNB

from sklearn.calibration import CalibratedClassifierCV
train_df.head()
train_df = train_df.drop(['game_season_shift'], axis = 1)
train_X.loc[:,'game_season'] = train_X['game_season'].astype(str)
test_X.loc[:,'game_season'] = test_X['game_season'].astype(str)
train_X.loc[:,'area_of_shot'] = train_X['area_of_shot'].astype(str)
test_X.loc[:,'area_of_shot'] = test_X['area_of_shot'].astype(str)
train_X.loc[:,'type_of_shot'] = train_X['type_of_shot'].astype(str)

test_X.loc[:,'type_of_shot'] = test_X['type_of_shot'].astype(str)
le_game_season = LabelEncoder()

le_area_of_shot = LabelEncoder()

le_type_of_shot = LabelEncoder()

train_X.loc[:,'game_season'] = le_game_season.fit_transform(train_X['game_season'])

test_X.loc[:,'game_season'] = le_game_season.transform(test_X['game_season'])

train_X.loc[:,'area_of_shot'] = le_area_of_shot.fit_transform(train_X['area_of_shot'])

test_X.loc[:,'area_of_shot'] = le_area_of_shot.transform(test_X['area_of_shot'])

train_X.loc[:,'type_of_shot'] = le_type_of_shot.fit_transform(train_X['type_of_shot'])

test_X.loc[:,'type_of_shot'] = le_type_of_shot.transform(test_X['type_of_shot'])
train_X = train_X.drop(['game_season_shift'], axis = 1)

test_X = test_X.drop(['game_season_shift'], axis = 1)
test_X = test_X.drop(['is_goal'], axis = 1)
train_X = train_X.drop(['is_goal'], axis = 1)

train_X = train_X.drop(['shot_id_number'], axis = 1)
train_X.head()
test_X.head()
train_X.shape
clf = GaussianNB()

clf.fit(train_X, train_y)

y_pred = clf.predict_proba(test_X)
result = pd.concat((submission['shot_id_number'], pd.DataFrame(y_pred)), axis = 1)
result.to_csv('ronaldo_pred', index=False)