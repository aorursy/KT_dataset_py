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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

import matplotlib.pyplot as plt

import seaborn as sns

import gc

from tqdm import tqdm_notebook # 오래 걸리는 작업 진행확인용
import warnings

warnings.filterwarnings(action='ignore')
pd.options.display.max_columns = 60
match = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/matchpre.pkl")



win_team_stat = pd.read_csv("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/win_team_stats.csv")



lose_team_stat = pd.read_csv("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/lose_team_stats.csv")



win_team = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/match_winner_data.pkl")



lose_team = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/match_lose_data.pkl")



date = pd.read_csv("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/lol_version_Date.csv")
display(match,win_team,lose_team,win_team_stat,lose_team_stat)
"""

Win, lose team 구분 확인

"""

for i in tqdm_notebook(range(len(win_team))):

    wf_valid = win_team['win'].iloc[i]

    

    if lose_team['win'].iloc[i] != wf_valid:

        pass

    else:

        print(str(i)+'데이터 정합성 문제')
"""

팀 스탯 row 와 팀 기록 row 일치시키기

"""

gameId = win_team_stat["gameId"]

match = pd.merge(gameId,match,how="inner",on="gameId")

win_team = pd.merge(gameId,win_team,how="inner",on="gameId")

lose_team = pd.merge(gameId,lose_team,how="inner",on="gameId")

display(match.shape,win_team.shape,lose_team.shape,win_team_stat.shape,lose_team_stat.shape)
"""

duplicated() 함수는 리스트에 대한 중복검사를 지원하지 않으므로 리스트 칼럼 삭제

"""

match.drop("participants",axis=1,inplace=True)



win_team.drop("bans",axis=1,inplace=True)



lose_team.drop("bans",axis=1,inplace=True)
display(match.shape,win_team.shape,lose_team.shape,win_team_stat.shape,lose_team_stat.shape)
display(match.duplicated().sum(),win_team.duplicated().sum(),lose_team.duplicated().sum())
display(win_team_stat.duplicated().sum(),lose_team_stat.duplicated().sum())
match = match.drop_duplicates()

win_team = win_team.drop_duplicates()

lose_team = lose_team.drop_duplicates()



win_team_stat = win_team_stat.drop_duplicates()

lose_team_stat = lose_team_stat.drop_duplicates()
display(match.shape,win_team.shape,lose_team.shape,win_team_stat.shape,lose_team_stat.shape)
"""

win_team, lose_team으로 나누어 모든 테이블 병합

"""

win_team = pd.merge(match,win_team,how="left",on="gameId")

win_team = pd.merge(win_team,win_team_stat,how="left",on="gameId")



lose_team = pd.merge(match,lose_team,how="left",on="gameId")

lose_team = pd.merge(lose_team,lose_team_stat,how="left",on="gameId")

display(win_team,lose_team)
del match

gc.collect()
"""

팀 데이터셋을 전체 데이터로 병합하기 전에 칼럼명 일치시키기

"""

win_team.columns = win_team.columns.str.replace("win_","")



lose_team.columns = lose_team.columns.str.replace("lose_","")
"""

전체 게임 정보 gamedata 테이블 생성

"""

gamedata = pd.concat([win_team,lose_team])

gamedata = gamedata.reset_index()

gamedata.drop("index",axis=1,inplace=True)

gamedata = gamedata.astype({"gameVersion":int})

gamedata = pd.merge(gamedata,date,how="inner",on="gameVersion")
del win_team

del lose_team

gc.collect()
gamedata
"""

카테고리형 데이터(True,False) Label encoding / 분석에 용이하도록 가공

"""

bool_mapping = {True:1,False:0}

bool_col = gamedata.select_dtypes('bool').columns.tolist()



for col in bool_col:

    gamedata[col] = gamedata[col].map(bool_mapping)

    

win_mapping = {"Win":1,"Fail":0}

gamedata["win"] = gamedata["win"].map(win_mapping)



team_mapping = {100:"Blue",200:"Red"}

gamedata["teamId"] = gamedata["teamId"].map(team_mapping)



gamedata["date"] = pd.to_datetime(gamedata["date"])

gamedata["gameId"] = gamedata.astype({"gameId":object})
gamedata
gamedata.info()
gamedata.describe()
gamedata.drop(["dominionVictoryScore","vilemawKills"],axis=1,inplace=True)
display(gamedata[gamedata['win']==1].describe(),gamedata[gamedata['win']==0].describe())
gamedata["date"].describe()
gameoverview = gamedata.drop('gameVersion',axis=1)
plt.figure(figsize=(20,12))

gameoverview.boxplot(vert=0)
"""

각 변수별 상관계수 시각화

"""

corr = gamedata.corr()



sns.clustermap(corr,cmap = 'RdYlBu_r', vmin = -1, vmax = 1)
gamedata["year"] = gamedata["date"].dt.year

gamedata["month"] = gamedata["date"].dt.month

gamedata["day"] = gamedata["date"].dt.day
f,ax = plt.subplots(3,1,figsize=(20,12))

sns.countplot(gamedata["year"],ax=ax[0])

sns.countplot(gamedata["month"],ax=ax[1])

sns.countplot(gamedata["day"],ax=ax[2])
#gameDuration 은 게임시작시부터 끝까지 초단위로 기록되어있다. 

gamedata["gameMinute"] = gamedata["gameDuration"] / 60
plt.figure(figsize=(20,12))

sns.distplot(gamedata["gameMinute"],kde=True)
gamedata = gamedata[gamedata["gameMinute"] > 5]
gamedata[(gamedata["gameMinute"] < 15)]
"""

해당 게임의 blue,red팀의 정보를 확인하기 위해 gameId를 라벨인코딩(조건문에서 gameId 값이 인식안됨)

"""

from sklearn.preprocessing import LabelEncoder

gamedata["gameId"] = LabelEncoder().fit_transform(gamedata["gameId"])

gamedata["gameId"].value_counts()
earlygame = gamedata[(gamedata["gameMinute"] < 15)]

earlygame
display(earlygame[earlygame["gameId"]==88419],earlygame[earlygame["gameId"]==87909],earlygame[earlygame["gameId"]==88253])
display(earlygame[earlygame['win']==1].describe(),earlygame[earlygame['win']==0].describe())
columns = gamedata.columns.tolist()
"""

게임 기록 win,lose 나눠서 시각화(빨강-lose, 파랑-win)

"""

f, axes = plt.subplots(20,2,figsize=(10,20))

early_win = earlygame[earlygame["win"]==1]

early_lose = earlygame[earlygame["win"]==0]



ax = axes.ravel()



for i,j in zip(range(5,45),range(0,40)):

    _,bins = np.histogram(earlygame.iloc[:,i],bins=50)

    ax[j].hist(early_win.iloc[:,i],bins=bins,color="blue",alpha=0.5)

    ax[j].hist(early_lose.iloc[:,i],bins=bins,color="red",alpha=0.5)

    ax[j].set_title(columns[i])

    ax[j].set_yticks(())

f.tight_layout()
game15_20 = gamedata[(gamedata["gameMinute"] < 20)&(gamedata["gameMinute"]>=15)]

game15_20
display(game15_20[game15_20['win']==1].describe(),game15_20[game15_20['win']==0].describe())
"""

게임 기록 win,lose 나눠서 시각화(빨강-lose, 파랑-win)

"""

f, axes = plt.subplots(20,2,figsize=(10,20))

win_15_20 = game15_20[game15_20["win"]==1]

lose_15_20 = game15_20[game15_20["win"]==0]



ax = axes.ravel()



for i,j in zip(range(5,45),range(0,40)):

    _,bins = np.histogram(game15_20.iloc[:,i],bins=50)

    ax[j].hist(win_15_20.iloc[:,i],bins=bins,color="blue",alpha=0.5)

    ax[j].hist(lose_15_20.iloc[:,i],bins=bins,color="red",alpha=0.5)

    ax[j].set_title(columns[i])

    ax[j].set_yticks(())

f.tight_layout()
game20_25 = gamedata[(gamedata["gameMinute"] < 25)&(gamedata["gameMinute"]>=20)]

game20_25
display(game20_25[game20_25['win']==1].describe(),game20_25[game20_25['win']==0].describe())
"""

게임 기록 win,lose 나눠서 시각화(빨강-lose, 파랑-win)

"""

f, axes = plt.subplots(20,2,figsize=(10,20))

win_20_25 = game20_25[game20_25["win"]==1]

lose_20_25 = game20_25[game20_25["win"]==0]



ax = axes.ravel()



for i,j in zip(range(5,45),range(0,40)):

    _,bins = np.histogram(game20_25.iloc[:,i],bins=50)

    ax[j].hist(win_20_25.iloc[:,i],bins=bins,color="blue",alpha=0.5)

    ax[j].hist(lose_20_25.iloc[:,i],bins=bins,color="red",alpha=0.5)

    ax[j].set_title(columns[i])

    ax[j].set_yticks(())

f.tight_layout()
game20_25.pivot_table(index="win", columns="firstBaron", aggfunc="size")
game25_30 = gamedata[(gamedata["gameMinute"] < 30)&(gamedata["gameMinute"]>=25)]

game25_30
display(game25_30[game25_30['win']==1].describe(),game25_30[game25_30['win']==0].describe())
"""

게임 기록 win,lose 나눠서 시각화(빨강-lose, 파랑-win)

"""

f, axes = plt.subplots(20,2,figsize=(10,20))

win_25_30 = game25_30[game25_30["win"]==1]

lose_25_30 = game25_30[game25_30["win"]==0]



ax = axes.ravel()



for i,j in zip(range(5,45),range(0,40)):

    _,bins = np.histogram(game25_30.iloc[:,i],bins=50)

    ax[j].hist(win_25_30.iloc[:,i],bins=bins,color="blue",alpha=0.5)

    ax[j].hist(lose_25_30.iloc[:,i],bins=bins,color="red",alpha=0.5)

    ax[j].set_title(columns[i])

    ax[j].set_yticks(())

f.tight_layout()
game25_30.pivot_table(index="win", columns="firstBaron", aggfunc="size")
game30_35 = gamedata[(gamedata["gameMinute"] < 35)&(gamedata["gameMinute"]>=30)]

game30_35
display(game30_35[game30_35['win']==1].describe(),game30_35[game30_35['win']==0].describe())
"""

게임 기록 win,lose 나눠서 시각화(빨강-lose, 파랑-win)

"""

f, axes = plt.subplots(20,2,figsize=(10,20))

win_30_35 = game30_35[game30_35["win"]==1]

lose_30_35 = game30_35[game30_35["win"]==0]



ax = axes.ravel()



for i,j in zip(range(5,45),range(0,40)):

    _,bins = np.histogram(game30_35.iloc[:,i],bins=50)

    ax[j].hist(win_30_35.iloc[:,i],bins=bins,color="blue",alpha=0.5)

    ax[j].hist(lose_30_35.iloc[:,i],bins=bins,color="red",alpha=0.5)

    ax[j].set_title(columns[i])

    ax[j].set_yticks(())

f.tight_layout()
late_game = gamedata[(gamedata["gameMinute"] >= 35)]

late_game
"""

게임 기록 win,lose 나눠서 시각화(빨강-lose, 파랑-win)

"""

f, axes = plt.subplots(20,2,figsize=(10,20))

late_win = late_game[late_game["win"]==1]

late_lose = late_game[late_game["win"]==0]



ax = axes.ravel()



for i,j in zip(range(5,45),range(0,40)):

    _,bins = np.histogram(late_game.iloc[:,i],bins=50)

    ax[j].hist(late_win.iloc[:,i],bins=bins,color="blue",alpha=0.5)

    ax[j].hist(late_lose.iloc[:,i],bins=bins,color="red",alpha=0.5)

    ax[j].set_title(columns[i])

    ax[j].set_yticks(())

f.tight_layout()
#승패에 영향이 가는 상관계수

gamedata.corr()["win"]