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
import pickle #데이터 저장용

import json

import re 

import time

from pandas.io.json import json_normalize

from tqdm import tqdm_notebook # 오래 걸리는 작업 진행확인용

import requests
"""

Match data 를 가져오기 위해선 먼저 각 소환사의 아이디가 필요하다. 따라서 챌린저, 그랜드마스터, 마스터에 소속된 유저명단을 불러온다.

"""

# api_key = "Riot developer's api key"



# api_challenger = "https://kr.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key=" + api_key

# api_grandmaster = "https://kr.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5?api_key=" + api_key

# api_master = "https://kr.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5?api_key=" + api_key

# r = requests.get(api_challenger)

# league = pd.DataFrame(r.json())

# r = requests.get(api_grandmaster)

# league = pd.concat([league, pd.DataFrame(r.json())], axis=0)

# r = requests.get(api_master)

# league = pd.concat([league, pd.DataFrame(r.json())], axis=0)



# league.reset_index(inplace=True)

# league_entries = pd.DataFrame(dict(league["entries"])).T

# league = pd.concat([league, league_entries], axis=1)



# league = league.drop(["index", "queue", "name", "leagueId", "entries", "rank"], axis=1) #필요없는 칼럼 드랍
"""

불러온 유저정보의 summonerId를 이용해 소환사의 정보api에서 account_id를 불러온다

API는 초당 20번, 분당 50번으로 요청 제한이 있으므로 블랙리스트에 오르지 않도록 유의해서 데이터를 수집하고 파일로 저장.

"""

# league["account_id"] = np.nan # account_id 초기화

# for i, summoner_id in enumerate(league["summonerId"]):

#     #Summoner API에서 AccountId를 가져와 채워넣는다.

#     api_url = "https://kr.api.riotgames.com/lol/summoner/v4/summoners/" + summoner_id + "?api_key=" + api_key

#     r = requests.get(api_url)

#     while r.status_code!=200: # 오류를 리턴할 경우 지연하고 다시 시도

#         time.sleep(5)

#         r = requests.get(api_url)

#     account_id = r.json()["accountId"]

#     league.iloc[i, -1] = account_id



# league.to_csv("LeagueData.csv")
"""

소환사의 account_id 로부터 치뤄졌던 경기들의 목록인 Matchlist를 불러온다.

"""



# season = str(13) #13시즌에 치뤄진 경기를 불러온다.



# match_info_df = pd.DataFrame()

# for account_id in league_df["account_id"]:

#     api_url = "https://kr.api.riotgames.com/lol/match/v4/matchlists/by-account/" + account_id + \

#                   "?season=" + season + "&api_key=" + api_key

#     r = requests.get(api_url)

#     while r.status_code!=200: # 오류를 리턴할 경우 지연하고 다시 시도

#         time.sleep(5)

#         r = requests.get(api_url)

#     match_info_df = pd.concat([match_info_df, pd.DataFrame(r.json()["matches"])])

# match_info_df.to_csv("MatchInfoData.csv")
"""

Matchlist에서 얻은 경기 한 판의 키값인 gameId로 해당 판의 기록을 불러온다.

"""



# match_info_df = match_info_df.drop_duplicates("gameId") #중복 경기기록은 삭제



# match_df = pd.DataFrame()

# for game_id in match_info_df["gameId"]:

#     api_url = "https://kr.api.riotgames.com/lol/match/v4/matches/" + str(game_id) + "?api_key=" + api_key

#     r = requests.get(api_url)

#     while r.status_code!=200: # 오류를 리턴할 경우 지연하고 다시 시도

#         time.sleep(5)

#         r = requests.get(api_url)

#     r_json = r.json()

#     temp_df = pd.DataFrame(list(r_json.values()), index=list(r_json.keys())).T

#     match_df = pd.concat([match_df, temp_df])



# match_df.to_pickle("Match.pkl") # 파일로 저장
# from urllib.request import urlopen

# from bs4 import BeautifulSoup

# import urllib

# import csv

# import re
"""

해당 사이트의 html 테이블 구조가 이상해 find 메서드가 제대로 작동 안함에 따라

odd,even,date를 find_all 따로 한번에 수집후 병합

"""

# url = "https://whatpulse.org/app/league-of-legends"

# html = urlopen(url)

# obj = BeautifulSoup(html.read(),"html.parser")



# temp = list(obj.find_all("td",{"align":"center"}))



# date = []

# for i in range(len(temp)):

#     if(i%2==1):

#         datei = temp[i].text

#         date.append(datei)



# tempodd = list(obj.find_all("tr",{"class":"odd"}))

# tempeven = list(obj.find_all("tr",{"class":"even"}))



# tempodd2 = []

# for i in range(len(tempodd)):

#     tempi = tempodd[i].text

#     tempodd2.append(tempi)



# tempeven2 = []

# for i in range(len(tempeven)):

#     tempi = tempeven[i].text

#     tempeven2.append(tempi)



# version = []

# for i in range(len(tempodd2)):

#     version.append(tempodd2[i])

#     version.append(tempeven2[i])





# version_date =  pd.DataFrame(data={"version":version,"date":date})



# version_date['gameVersion'] = version_date['version'].str.replace(pat=r'[^0-9]', repl=r'', regex=True)



# version_date = version_date[["date","gameVersion"]]



# version_date.to_csv("lol_version_Date.csv",index=False)
"""

게임 한 판당 team red, blue의 기록

"""

# data = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/match.pkl")

# temp_team = list(data["teams"])

# #team red

# team1_df = pd.DataFrame()

# for i in tqdm_notebook(range(len(temp_team))):

#     temp_team[i][0].pop("bans",None)

#     team1 = pd.DataFrame(list(temp_team[i][0].values()),index = list(temp_team[i][0].keys())).T

#     team1_df = team1_df.append(team1)



# team1_df.index = range(len(team1_df))



# #team blue

# team2_df = pd.DataFrame()

# for i in tqdm_notebook(range(len(temp_team))):

#     temp_team[i][1].pop("bans",None)

#     team2 = pd.DataFrame(list(temp_team[i][1].values()),index = list(temp_team[i][1].keys())).T

#     team2_df = team2_df.append(team2)

    

# team2_df.index = range(len(team2_df))



# team red와 team blue, gameId 합치기

# data_team_df = pd.concat([team1_df,team2_df,data[["gameId"]]],axis=1)



# data_team_df.to_pickle("team.pkl") 
"""

팀 데이터 불러온 후, 팀 json 중간중간 비어있는 값 제외 

"""

# team_a_error = [] 

# team_b_error = []

# team_a = pd.DataFrame()

# team_b = pd.DataFrame()

# for i in range(len(lol_df)):

#     try:

#         team_a = team_a.append(json_normalize(lol_df["teams"].iloc[i][0]))

#         team_b = team_b.append(json_normalize(lol_df["teams"].iloc[i][1]))

#     except:

#         team_a_error.append(i)

#         team_b_error.append(i)

#         print(str(i)+"행에서 오류")

#         pass
"""

승리팀, 패배팀으로 나누고 정합성 확인

"""

# for i in range(len(team_a)):

#     wf_valid = team_a["win"].iloc[i]

    

#     if team_b["win"].iloc[i] != wf_valid:

#         pass

#     else:

#         print(str(i)+"행 정합성 문제")
""" 

게임 한판당 팀 구분 없는 모든 참가자(10*90500)명의 스탯 데이터

"""

# use_cols = ["kills","deaths","totalDamageDealtToChampions","goldEarned", "visionScore","totalTimeCrowdControlDealt"] Making columns to Using stat infomation/스탯에서 칼럼으로 추출할 것 기재 

# stats_df1 = pd.DataFrame()

# for i in tqdm_notebook(range(30000),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         cur_values = {f"{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(10)}

#         temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df1 = pd.concat([stats_df1, temp], axis=1, sort=False)

# stats_df1 = stats_df1.T.reset_index(drop=True)



# stats_df2 = pd.DataFrame()

# for i in tqdm_notebook(range(30000,60000),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         cur_values = {f"{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(10)}

#         temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df2 = pd.concat([stats_df2, temp], axis=1, sort=False)

# stats_df2 = stats_df2.T.reset_index(drop=True)



# stats_df2 = pd.concat([stats_df1,stats_df2], ignore_index=True)



# stats_df3 = pd.DataFrame()

# for i in tqdm_notebook(range(60000,90500),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         cur_values = {f"{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(10)}

#         temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df3 = pd.concat([stats_df3, temp], axis=1, sort=False)

# stats_df3 = stats_df3.T.reset_index(drop=True)



# stats_df = pd.concat([stats_df,stats_df3], ignore_index=True)



# stats_df["gameId"] = match["gameId"]



# stats_df.info()



# stats_df.to_csv("stats.csv",index=False)
"""

게임 한 판당 승리한 팀원들의 스탯

"""

# use_cols = ["kills","deaths","totalDamageDealtToChampions","goldEarned", "visionScore","totalTimeCrowdControlDealt"]

# stats_df = pd.DataFrame()

# for i in tqdm_notebook(range(30000),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         if (match["participants"].iat[i][0]["stats"]["win"]):

#             cur_values = {f"win_{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(5)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#         else:

#             cur_values = {f"win_{col}{j-4}": match["participants"].iat[i][j]["stats"][col] for j in range(5,10)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df = pd.concat([stats_df, temp], axis=1, sort=False)

# stats_df = stats_df.T.reset_index(drop=True)



# stats_df2 = pd.DataFrame()

# for i in tqdm_notebook(range(30000,60000),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         if (match["participants"].iat[i][0]["stats"]["win"]):

#             cur_values = {f"win_{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(5)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#         else:

#             cur_values = {f"win_{col}{j-4}": match["participants"].iat[i][j]["stats"][col] for j in range(5,10)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df2 = pd.concat([stats_df2, temp], axis=1, sort=False)

# stats_df2 = stats_df2.T.reset_index(drop=True)



# win_stats_df = pd.concat([stats_df,stats_df2], ignore_index=True)



# stats_df3 = pd.DataFrame()

# for i in tqdm_notebook(range(60000,90500),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         if (match["participants"].iat[i][0]["stats"]["win"]):

#             cur_values = {f"win_{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(5)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#         else:

#             cur_values = {f"win_{col}{j-4}": match["participants"].iat[i][j]["stats"][col] for j in range(5,10)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df3 = pd.concat([stats_df3, temp], axis=1, sort=False)

# stats_df3 = stats_df3.T.reset_index(drop=True)



# win_stats_df = pd.concat([win_stats_df,stats_df3], ignore_index=True)



# win_stats_df["gameId"] = match["gameId"]



# win_stats_df.info()



# win_stats_df.to_csv("win_team_stats.csv",index=False)
"""

게임 한 판당 패배한 팀원들의 스탯

"""

# use_cols = ["kills","deaths","totalDamageDealtToChampions","goldEarned", "visionScore","totalTimeCrowdControlDealt"]

# stats_df1 = pd.DataFrame()

# for i in tqdm_notebook(range(30000),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         if not(match["participants"].iat[i][0]["stats"]["win"]):

#             cur_values = {f"lose_{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(5)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#         else:

#             cur_values = {f"lose_{col}{j-4}": match["participants"].iat[i][j]["stats"][col] for j in range(5,10)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df1 = pd.concat([stats_df1, temp], axis=1, sort=False)

# stats_df1 = stats_df1.T.reset_index(drop=True)



# stats_df2 = pd.DataFrame()

# for i in tqdm_notebook(range(30000,60000),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         if not(match["participants"].iat[i][0]["stats"]["win"]):

#             cur_values = {f"lose_{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(5)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#         else:

#             cur_values = {f"lose_{col}{j-4}": match["participants"].iat[i][j]["stats"][col] for j in range(5,10)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df2 = pd.concat([stats_df2, temp], axis=1, sort=False)

# stats_df2 = stats_df2.T.reset_index(drop=True)



# lose_stats_df = pd.concat([stats_df1,stats_df2], ignore_index=True)



# stats_df3 = pd.DataFrame()

# for i in tqdm_notebook(range(60000,90500),desc="total"):

#     temp = pd.DataFrame()

#     for col in use_cols:

#         if not(match["participants"].iat[i][0]["stats"]["win"]):

#             cur_values = {f"lose_{col}{j+1}": match["participants"].iat[i][j]["stats"][col] for j in range(5)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#         else:

#             cur_values = {f"lose_{col}{j-4}": match["participants"].iat[i][j]["stats"][col] for j in range(5,10)}

#             temp = pd.concat([temp, pd.Series(cur_values)], axis=0, sort=False)

#     stats_df3 = pd.concat([stats_df3, temp], axis=1, sort=False)

# stats_df3 = stats_df3.T.reset_index(drop=True)



# lose_stats_df = pd.concat([lose_stats_df,stats_df3], ignore_index=True)



# lose_stats_df["gameId"] = match["gameId"]



# lose_stats_df.info()



# lose_stats_df.to_csv("lose_team_stats.csv",index=False)
"""

match 데이터가 차지하는 램 용량이 너무 커서 전처리

"""

# match = pd.read_pickle("/kaggle/input/lol-classic-rank-game-datakrtop-3-tier/match_ver1.pkl")

# match = match[match["gameMode"] == "CLASSIC"]

# match.drop(["gameCreation","gameMode","gameType","mapId","platformId","status.message","status.status_code"],axis=1,inplace=True)

# match['gameVersion'] = match['gameVersion'].str.split(".").map(lambda x: "".join(x))

# match.to_pickle("matchpre.pkl")