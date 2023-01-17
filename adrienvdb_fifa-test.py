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
df_fifa = pd.read_csv("../input/WorldCupMatches.csv")
print("Loaded")
df_fifa.head()
df_fifa.drop(columns=['Referee', 'Win conditions', 'Assistant 1', 'Assistant 2', 'RoundID', 'MatchID', 'Home Team Initials', 'Away Team Initials'], inplace = True)
%matplotlib inline

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
df_attendance = pd.DataFrame(df_fifa.groupby("Year")[['Attendance']].sum())
df_attendance.reset_index(inplace=True)
df_attendance.head()
df_attendance.plot(kind="scatter", x='Year', y='Attendance')
plt.show()
df_fifa.head()
df_fifa["Winner"] = 'Draw'
for index, row in df_fifa.iterrows():
    if row['Home Team Goals']>row['Away Team Goals']:
        df_fifa.at[index, 'Winner'] = row['Home Team Name']
    elif row['Home Team Goals']<row['Away Team Goals']:
        df_fifa.at[index,'Winner'] = row['Away Team Name']
df_france = pd.DataFrame(df_fifa[df_fifa['Winner'] == 'France'].groupby("Year")['Winner'].count())
df_france.reset_index(inplace=True)
df_france.plot(kind='scatter', x='Year', y='Winner')
plt.show()
df_final = df_fifa[df_fifa['Stage']=='Final'].groupby("Winner").size().reset_index(name='counts')
df_final.sort_values(by='counts', inplace=True)
df_final.drop(2, inplace=True) #drop the 'Draw' winner since not relevant
df_final.plot.bar(x='Winner',figsize=(20, 7))
plt.show()
