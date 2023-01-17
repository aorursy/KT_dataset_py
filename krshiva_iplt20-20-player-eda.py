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
ipl_batsman = pd.read_csv('/kaggle/input/ipl2020-players-dataset/batsmen.csv')
ipl_bowler = pd.read_csv('/kaggle/input/ipl2020-players-dataset/bowlers.csv')
ipl_allround = pd.read_csv('/kaggle/input/ipl2020-players-dataset/allrounders.csv')
#ipl_batsman.isnull().sum()
#ipl_allround.isnull().sum()
#ipl_bowler.isnull().sum()
ipl_bowler.info()
# datset is 63X15
ipl_allround.info()
# dataset is 47X24
ipl_batsman.info()
# datset is 75X 15 
# colums in ipl_batsan dataset
#ipl_batsman.columns
# Top five batsman by Average runs 

y=ipl_batsman.nlargest(3, ['Bat_Avg'])
print(y[['Name_Of_Player', 'Bat_Avg','Team']])
# Top five batsman by strike rate  
y=ipl_batsman.nlargest(3, ['50s'])
print(y[['Name_Of_Player', '50s', 'Team']])
# most 50s by a batsman 
y=ipl_batsman.nlargest(3, ['Bat_SR'])
print(y[['Name_Of_Player', 'Bat_SR', 'Team']])
ipl_bowler.columns
# Bowler with most wickets 
y=ipl_bowler.nlargest(3, ['Wickets'])
print(y[['Name_Of_Player', 'Wickets','Team']])

## Bowler with least run conceaded who played at least 14 matchs

if ipl_bowler[['Matches']]>14:
    y=ipl_bowler.nsmallest(3, ['Runs_Conceded'])
    print(y[['Name_Of_Player', 'Runs_Conceded','Team', 'Matches']])
# Bowler with best wickets 
y=ipl_bowler.nsmallest(3, ['Eco'])
print(y[['Name_Of_Player', 'Eco','Team']])


