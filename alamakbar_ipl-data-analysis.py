# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Task 1:Read CSV Files
df_deliveries = pd.read_csv("/kaggle/input/ipl/deliveries.csv")
df_matches = pd.read_csv("/kaggle/input/ipl/matches.csv")
#Task 2: Get top-level summary
df_deliveries.head()
df_matches.head()
df_deliveries.columns
df_matches.columns
df_deliveries.info()
df_matches.info()
df_deliveries.shape
df_matches.shape
df_deliveries.nunique()
df_matches.nunique()
df_deliveries.describe()
df_matches.describe()
#Task 3: Check if column 'umpire3' is useful. If not drop it and save the dataframe to a file.
df_matches['umpire3'].isnull().sum()
#so,the umpire3 column is complete empty with no values so we must drop it.
df_matches.drop(['umpire3'], axis = 1, inplace=True)
df_matches.head() #we can see the 'umpire3' column has been dropped which was of no use
#Task 4: Print Total Number of Mataches played, Number of Seasons
print("Total Number of Matches:", df_matches.shape[0])
print("Total Number of Seasons:" , df_matches['season'].nunique())
#Task 5: Players who have won the most "Player of the Match" 
#And Team which has the highest number of match wins

print("Won the most Player of the Match: ", df_matches['player_of_match'].value_counts().idxmax())
print(df_matches['winner'].value_counts().idxmax()+",", "has the most most match wins")
#Task 6: Find Teams which have won matches by a huge margin
huge_margin = df_matches[(df_matches['win_by_runs']>=100) | (df_matches['win_by_wickets']>=8)]
huge_margin.winner.value_counts()
#Task 7: Find Count of Matches Played at each city
df_matches['city'].value_counts()
#Task 8: Find Min and Max win_by_wickets at each city
df_matches.groupby('city').win_by_wickets.agg(['min', 'max'])
#Task 9: Find the match with biggest defeat by runs,
df_matches.groupby(['winner']).win_by_runs.min()
#Task 10: Find the match with biggest defeat by wickets
df_matches.groupby(['winner']).win_by_wickets.min()