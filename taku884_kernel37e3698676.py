# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/train.csv')

df_test = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/test.csv')
df_train
df_test
df_train.shape
df_test.shape
df_train.columns
df_train.describe()
df_train = df_train.fillna(0)

df_test = df_test.fillna(0)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['player_positions','preferred_foot']:

  df_train[col] = le.fit_transform(df_train[col])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['player_positions','preferred_foot']:

  df_test[col] = le.fit_transform(df_test[col])
df_train
colmuns = ['id','age', 'height_cm', 'weight_kg', 'overall', 'potential',

           'player_positions', 'preferred_foot', 'weak_foot',

           'skill_moves','defending_marking','defending_standing_tackle', 'goalkeeping_diving', 'goalkeeping_handling',

           'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes',

           'pace', 'shooting', 'passing', 'dribbling', 'physic', 'attacking_crossing', 'attacking_short_passing',

           'attacking_volleys', 'skill_dribbling','skill_curve', 'skill_fk_accuracy', 'skill_long_passing',

           'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed',

           'movement_agility', 'movement_reactions', 'movement_balance',

           'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',

           'power_long_shots','defending_sliding_tackle']
X_train = df_train[colmuns]

y_train = df_train['value_eur']

X_test = df_test[colmuns]
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-football-players-value-prediction/sampleSubmission.csv', index_col=0)

submit_df['value_eur'] = p_test

submit_df
submit_df.to_csv('submission.csv')