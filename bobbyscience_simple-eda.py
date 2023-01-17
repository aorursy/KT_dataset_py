import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline
X_df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')
X_df.head()
X_df.info()
X_df.describe()
# We don't care about gameId there.
X_df = X_df.drop(columns=['gameId'])
X_df['blueWins'].value_counts()
sns.countplot(X_df['blueWins'], palette='plasma')
X_df.hist(bins=50, figsize=(20, 15))
plt.show()
corr_matrix = X_df.corr()
corr_matrix['blueWins'].sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr_matrix, ax=ax, cmap='plasma')
plt.show()
attrs = [
    'blueGoldDiff',
    'blueExperienceDiff',
    'blueGoldPerMin',
    'blueTotalGold',
    'blueTotalExperience',
    'blueKills',
]

scatter_matrix(X_df[attrs], figsize=(12, 8))
plt.show()
labels = X_df['blueWins']
labels.shape
features = X_df.drop(columns=['blueWins'])
features.shape