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
url = 'https://www.kaggleusercontent.com/episodes/3634538.json'
import urllib, json



response = urllib.request.urlopen(url)

data = json.loads(response.read())
dict_obs = {}

for i in range(3002):

    dict_obs[i] = data['steps'][i][0]['observation']['players_raw'][0] # left player

    dict_obs[i]['action'] = data['steps'][i][0]['action'][0] if len(data['steps'][i][0]['action']) > 0 else -1

    #dict_obs[i] = data['steps'][i][1]['observation']['players_raw'][0] # right player
df = pd.DataFrame(dict_obs)

df = df.T

df
df.describe()
import matplotlib.pyplot as plt
df['action'].plot()
df['action'].value_counts().plot(kind='barh')
df['ball_owned_team'].plot()
df['ball_owned_team'].value_counts().plot(kind='barh')
df['ball_owned_player'].plot()
df['ball_owned_player'].value_counts().plot(kind='barh')
df['active'].plot()
df['active'].value_counts().plot(kind='barh')
df['game_mode'].plot()
df['game_mode'].value_counts().plot(kind='barh')