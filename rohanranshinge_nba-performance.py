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


import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
full_nba = pd.read_csv("../input/nba_2016_2017_100.csv")
attendance = pd.read_csv('../input/nba_2017_att_val.csv')
endorsements = pd.read_csv('../input/nba_2017_endorsements.csv')
full_nba.head(3)
full_nba[['PLAYER_NAME','TEAM_ABBREVIATION','AGE','W_PCT','MIN','NET_RATING','PIE','FG_PCT','SALARY_MILLIONS','PTS','TWITTER_FOLLOWER_COUNT_MILLIONS']].sort_values(by='SALARY_MILLIONS',ascending=False)
nba = full_nba[['PLAYER_NAME','TEAM_ABBREVIATION','AGE','W_PCT','MIN','NET_RATING','PIE','FG_PCT','SALARY_MILLIONS','PTS','TWITTER_FOLLOWER_COUNT_MILLIONS']].sort_values(by='TWITTER_FOLLOWER_COUNT_MILLIONS',ascending=False)
attendance.head(3)
len(attendance)
#attendance[].corr()
attendance[['TOTAL','AVG','PCT','VALUE_MILLIONS']].corr()
attendance.head()
lm = smf.ols('VALUE_MILLIONS ~ TOTAL',data = attendance).fit()
print(lm.summary())
nba
clustering = nba[["W_PCT", "MIN", "NET_RATING", "PIE","SALARY_MILLIONS","PTS","TWITTER_FOLLOWER_COUNT_MILLIONS"]]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(clustering))
print(scaler.transform(clustering))
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(clustering))
nba['CLUSTER'] = kmeans.labels_

#nba.drop(['cluster'],axis=1,inplace=True)
nba['CLUSTER'] += 1
nba.sort_values(by='CLUSTER')
top_performers = nba[nba.CLUSTER==0]
top_performers
nba.tail()
for i in range(1,4):
   # plt.figure(figsize=(5,5))
    plt.subplots(figsize=(15,5))
    plt.subplot(1, 3, i)
    if i ==1:
        plt.scatter( nba.loc[nba['CLUSTER'] == i]['SALARY_MILLIONS'],nba.loc[nba['CLUSTER'] == i]['PTS'],color='darkgreen', alpha=0.5)
        plt.xlabel('SALARY')
        plt.ylabel('PTS')
        plt.title('CLUSTER %d' %i)
    elif i == 2:
        plt.scatter( nba.loc[nba['CLUSTER'] == i]['SALARY_MILLIONS'],nba.loc[nba['CLUSTER'] == i]['PTS'], alpha=0.5)
        plt.xlabel('SALARY')
        plt.ylabel('PTS')
        plt.title('CLUSTER %d' %i)
    elif i == 3:
        plt.scatter( nba.loc[nba['CLUSTER'] == i]['SALARY_MILLIONS'],nba.loc[nba['CLUSTER'] == i]['PTS'], color='red', alpha=0.5)
        plt.xlabel('SALARY')
        plt.ylabel('PTS')
        plt.title('CLUSTER %d' %i)
        
plt.tight_layout()

