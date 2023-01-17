# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv")
df.head()
df.info()
df.describe()
df.corr()
import seaborn as sns
sns.heatmap(df.corr()) 
df[['blueWins','blueDragons']].sum()
print(f"Red Wins: {df[df.blueWins == 0]['blueWins'].count()}");
print(f"Red Dragons: {df['redDragons'].sum()}")
blue = df[['blueDragons','blueWins']].sum()
red = pd.Series([df['redDragons'].sum(),df[df.blueWins == 0]['blueWins'].count()] ,index = ['redDragons', 'redWins'])
pd.crosstab(df['blueDragons'], df['blueWins'])
print(red); print(blue)
print(f"The Red Team Had More '{red.redDragons - blue.blueDragons}' Dragons More Than The Blue Team"); 
print("However, Getting The Dragon Every Match Won't Lead Your Team Into Winning!");
print(f"Red Team Won '{red.redWins - blue.blueWins}' Matches More Than The Blue Team")
df.head() 