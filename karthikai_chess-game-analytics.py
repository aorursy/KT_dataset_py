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
df=pd.read_csv("/kaggle/input/chess/games.csv")
df.head()
df.shape
len(df.id.unique())
df.isna().sum()
win_counts=df.winner.value_counts()
white=(win_counts["white"]/(len(df.winner)))*100

black=(win_counts["black"]/(len(df.winner)))*100

draw=(win_counts["draw"]/(len(df.winner)))*100

print("white winning percentage = "+str(white)+"%")

print("black winning percentage = "+str(black)+"%")

print("draw percentage = "+str(draw)+"%")
opening=df.groupby(by="opening_name").winner.value_counts()

print(opening)
opening=opening.reset_index(name="wins")
opening
opening.sort_values(by="wins",ascending=False).head()
black_winner=opening[opening.winner=="black"]

white_winner=opening[opening.winner=="white"]

draw_game=opening[opening.winner=="draw"]
black_winner.sort_values(by="wins",ascending=False).head()
white_winner.sort_values(by="wins",ascending=False).head()
df.groupby(by="victory_status").winner.value_counts()
df.describe(include="all")
opening_eco_analysis=df.groupby("winner").opening_eco.value_counts()

opening_eco_analysis=opening_eco_analysis.reset_index(name="count")

opening_eco_analysis
black_opening_eco=opening_eco_analysis[opening_eco_analysis.winner=="black"]

black_head=black_opening_eco.sort_values(ascending=False,by="count").head()

black_head
white_opening_eco=opening_eco_analysis[opening_eco_analysis.winner=="white"]

white_head=white_opening_eco.sort_values(ascending=False,by="count").head()

white_head
white_prob=[]

for i,j in enumerate(white_head["count"]):

    prob=j/1934

    white_prob.append(prob)

    

black_prob=[]

for i,j in enumerate(black_head["count"]):

    prob=j/2006

    black_prob.append(prob)

        
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

plt.figure(figsize=(10,7))

plt.xticks(rotation='vertical')

sns.countplot(df.opening_eco.sort_values(ascending=False).head(20),hue=df.winner)

white_head["prob"]=white_prob

black_head["prob"]=black_prob
sns.barplot(white_head.opening_eco,white_head.prob)
sns.barplot(black_head.opening_eco,black_head.prob)