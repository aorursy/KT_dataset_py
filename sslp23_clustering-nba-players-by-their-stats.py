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
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
df = pd.read_csv('/kaggle/input/basketball-players-stats-per-season-49-leagues/players_stats_by_season_full_details.csv')

df.shape


df_nba = df[(df['League'] == 'NBA') & (df['Season'] == '2018 - 2019') & (df['Stage'] == 'Regular_Season')].reset_index(drop=True)

df_nba.head()
#Number of players

print(len(df_nba.Player))

#Columns available

print(df_nba.columns.values)
#Columns that we will use to cluster

columns =  ['Player', 'GP', 'MIN', 'FGM',

       'FGA', '3PM', '3PA', 'FTM', 'FTA', 'TOV', 'PF', 'ORB', 'DRB',

       'REB', 'AST', 'STL', 'BLK', 'PTS']

df_nba_ = df_nba[columns]
#Some EDA...

#function used to find outliers based in quartiles

def find_outliers(data):

    data = sorted(data)

    Q1, Q3 = np.percentile(data, [25,75])

    IQR = Q3-Q1

    lower_bound = Q1 -(1.5 * IQR) 

    upper_bound = Q3 +(1.5 * IQR)

    return lower_bound, upper_bound



gp_lower, gp_upper = find_outliers(df_nba_['GP'])

gp_lower
df_filt = df_nba_[df_nba_['GP'] > gp_lower]

len(df_filt['Player'])
#df that will be used in model

#contains only stats and name of the player

df_p = df_filt[np.append(df_filt.columns.values[0], (df_filt.columns.values[3:]))].reset_index(drop=True)

df_p.head()
int_cols = df_p.columns.values[1:]



#correlation

plt.figure(figsize=(25,10))

cm = np.corrcoef(df_p[int_cols].values.T)

sns.set(font_scale=1.5)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=int_cols, xticklabels=int_cols)

plt.show()
X = df_p[int_cols]
sc = preprocessing.StandardScaler()

X_std = sc.fit_transform(X)

X_std
kmeans = KMeans(n_clusters=5, random_state=0)

kmeans.fit(X_std)
df_p['cluster'] = kmeans.predict(X_std)

df_p.head()
data_plot = df_p[['3PM', 'TOV', 'PF', 'REB', 'AST', 'STL', 'BLK', 'PTS', 'cluster']]

data_plot.groupby("cluster").aggregate("mean").plot.bar(figsize=(15,10))

plt.title("Stats by cluster")