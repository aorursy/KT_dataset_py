import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')
list(df.columns)
df.head()
len(df.columns)
df.info()
type(df.info())
df['GameWeather'].unique()
len(df['GameWeather'].unique())
num_variable = [

    'GameId', # 512 games

    'PlayId', # 23171 play

    'X',

    'Y',

    'S',

    'A',

    'Dis',

    'Orientation',

    'Dir',

    'NflId', # 2231 values

    # 'JerseyNumber', # 99 numbers

    

]



cat_variables = [

    'Team', # 2 values: Home or away

    'DisplayName', # 2230

    'Season', # two values: 2017, 2018

    'YardLine', # 50 values

    ''

]
import missingno as msno
msno.bar(df);
msno.matrix(df);
msno.matrix(df.sort_values('GameWeather'))
msno.heatmap(df)
msno.dendrogram(df);