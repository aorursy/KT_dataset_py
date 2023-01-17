import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('/kaggle/input/pga-tour-tournament-scoring-average-19802019/Scoring_Average_1980_2019.csv')

df_winner = df[df['Scoring Average - (RANK THIS WEEK)'] == 1]

df_winner['DATE'] = pd.to_datetime(df_winner['DATE'])

df_winner['year'] = df_winner['DATE'].dt.year

df_winner['year_rounded'] = df_winner['year'].round(-1) 

df_winner.groupby('year_rounded')['Scoring Average - (AVG)'].mean()[:-1]