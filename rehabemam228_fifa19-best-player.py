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
pd.set_option('display.max_columns',None)
df = pd.read_csv('../input/fifa19/data.csv', index_col=None)
df.head()
df.info()
# get subset that has skills and name
players_skills = df.iloc[:, np.r_[:3, 54:87]]
players_skills.head()
players_skills.columns
skill_col = ['Crossing', 'Finishing', 'HeadingAccuracy',
       'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy',
       'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility',
       'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength',
       'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision',
       'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle',
       'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning']
# make a dictionary that has the skill and player with max skill score
best_skills = {}
for col in skill_col:
    # best_skills[col] = players_skills.loc[players_skills[col].max()]['Name']
    players_skills.set_index('Name')
    best_skills[col] = df[players_skills[col] == players_skills[col].max()]['Name']
best_skills
# examine best players
best_skills['GKPositioning']
# get best player with highest max skills
skills = pd.DataFrame(best_skills)
skills.fillna(0, inplace=True)
skills.transpose().value_counts().idxmax()


    

