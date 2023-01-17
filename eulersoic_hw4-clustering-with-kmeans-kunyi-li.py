import seaborn as sns

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
players_salary = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_salary.csv')

players_stats = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_players_stats_combined.csv')

players_edsm = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_endorsements.csv')

players_twitter = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_twitter_players.csv')

players_RPM = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_real_plus_minus.csv')

team_value = pd.read_csv('/kaggle/input/social-power-nba/nba_2017_att_val.csv')
players_salary
players_stats = players_stats.rename(columns={'PLAYER':'NAME'})

players_stats
players_stats.info()
p_s = players_stats.drop(players_stats[["POSITION","TEAM","Unnamed: 0","Rk","AGE"]],axis=1)
pss = pd.merge(players_salary,p_s,how='inner',on='NAME')

pss = pss.drop(players_stats[["POSITION","TEAM","NAME"]],axis=1)
pss
pss.isnull().sum()
p_s.fillna(p_s["3P%"].mean(),inplace=True)
pss.fillna(p_s["3P%"].mean(),inplace=True)
pss.isnull().sum()
pss
ps = pss.drop("SALARY",axis=1)

ps
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3).fit(ps)

ps['cluster'] = km.labels_

ps.sort_values('cluster')

cluster_centers = km.cluster_centers_

ps.groupby("cluster").mean()
from pandas.plotting import scatter_matrix 

centers = ps.groupby("cluster").mean().reset_index

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10

colors = np.array(['red','green','blue','yellow'])

plt.scatter(ps["MP"],ps["WINS_RPM"],c=colors[ps["cluster"]])

plt.scatter(centers.MP, centers.WINS_RPM,linewidths=50,marker='+',s=3,c='black')

plt.xlabel("MP")

plt.ylabel("WINS_RPM")

scatter_matrix(ps[['GP','MPG','ORPM','DRPM','RPM','WINS_RPM']],s=50,alpha=1,c=colors[ps["cluster"]],figsize=(20,20))
km1 = KMeans(n_clusters=3).fit(pss)

pss['cluster'] = km1.labels_

pss.sort_values('cluster')

cluster_centers = km1.cluster_centers_

pss.groupby("cluster").mean()
from pandas.plotting import scatter_matrix 

centers = pss.groupby("cluster").mean().reset_index

%matplotlib inline

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 10

colors = np.array(['red','green','blue','yellow'])

plt.scatter(pss["SALARY"],pss["WINS_RPM"],c=colors[pss["cluster"]])

plt.scatter(centers.MP, centers.WINS_RPM,linewidths=50,marker='+',s=3,c='black')

plt.xlabel("SALARY")

plt.ylabel("WINS_RPM")
scatter_matrix(pss[['SALARY','GP','MPG','ORPM','DRPM','RPM','WINS_RPM']],s=50,alpha=1,c=colors[ps["cluster"]],figsize=(20,20))