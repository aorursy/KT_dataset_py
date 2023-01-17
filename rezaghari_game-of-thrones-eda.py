import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
eps = pd.read_csv('/kaggle/input/game-of-thrones/GOT_episodes_v4.csv')
eps.head()
eps.drop(['Summary', 'Budget_estimate'], axis=1, inplace=True)
eps.head()
season_rating = round(eps.groupby('Season').Rating.mean().to_frame(),2).reset_index()
season_rating.columns = ['Season', 'Rating']
plt.figure(figsize=(10,5))
sns.barplot(y=season_rating.Season, x=season_rating.Rating, orient='h')
plt.xlim(5.5,9.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Average Rating Per Season", fontsize=16)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Season', fontsize=14)
top_10_eps = eps[['Season', 'Episode','Title', 'Rating', 'Votes']].sort_values(['Rating','Votes'], ascending=False)[:10]
plt.figure(figsize=(10,5))
sns.barplot(y=top_10_eps.Title, x=top_10_eps.Rating, orient='h', palette='Greens_d')
plt.xlim(9,10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Top Ten episodes by Ratings", fontsize=15)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Episode Title', fontsize=14)
bottom_10_eps = eps[['Season', 'Episode','Title', 'Rating', 'Votes']].sort_values(['Rating','Votes'], ascending=True)[:10]
bottom_10_eps
plt.figure(figsize=(10,5))
sns.barplot(y=bottom_10_eps.Title, x=bottom_10_eps.Rating, orient='h', palette='Reds_d')
plt.xlim(3,9)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Bottom Ten episodes by Ratings", fontsize=15)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Episode Title', fontsize=14)
# eps.groupby('Director').value_counts()
dirs_eps = eps.Director.value_counts().to_frame().reset_index()
dirs_eps.columns = ['Director', 'Episodes']
dirs_rates = round(eps.groupby('Director').Rating.mean().to_frame().reset_index(),2)

dirs = dirs_eps[dirs_eps.Episodes >=4].merge(dirs_rates, how='left', on='Director')
dirs = dirs.sort_values('Rating', ascending=False)
plt.figure(figsize=(10,5))
sns.barplot(y=dirs.Director, x=dirs.Rating, orient='h', palette='Reds_d')
plt.xlim(8,9.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("Bottom Ten episodes by Ratings", fontsize=15)
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Episode Title', fontsize=14)