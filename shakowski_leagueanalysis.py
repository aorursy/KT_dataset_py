import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import json
import pandas as pd
import time
import plotly
import seaborn as sns
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#################Inputs
name = "sexualcuddler"
key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
#######################
def get_match_data(summoner_name, api_key):
    #Return ID of the person you want to look into
    url = "https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/"
    url += summoner_name + "?api_key=" + api_key
    data = requests.get(url).json()
    id = data["accountId"]
    
    #Return the match history of the player
    url = "https://na1.api.riotgames.com/lol/match/v4/matchlists/by-account/"
    url += id + "?api_key=" + api_key
    data = requests.get(url).json()
    game_IDs= [x["gameId"] for x in data["matches"]]
    
    #return the information for for each match in match history
    list_of_games = []
    
    for game_id in game_IDs:
        url = "https://na1.api.riotgames.com/lol/match/v4/matches/"
        url += str(game_id) + "?api_key=" + api_key
        list_of_games.append(requests.get(url).json())
        
    #print this information to a text file for future use.
    open('Output.txt', 'w').close()
    with open("Output.txt", "w", encoding="utf-8") as f:
        for game in list_of_games:
            f.write("%s\n" % json.dumps(game))           
# get_match_data(name, key) 
def read_text_file():
    with open("/kaggle/input/leaguedata/Output.txt") as f:
        content = f.readlines()

    return [json.loads(x) for x in content]

def return_participant_ID(dict, name):
    
    for x in dict["participantIdentities"]:
        if x["player"]["summonerName"] == name:
            return (x["participantId"])

def get_stats(dict, ID):
    
    for x in dict["participants"]:
        if x["participantId"] == ID:
            stat_dict = x["stats"]
            stat_dict["championId"] = x["championId"]
            stat_dict["spell1Id"] = x["spell1Id"]
            stat_dict["spell2Id"] = x["spell2Id"]
            stat_dict["game mode"] = dict["gameMode"]
            stat_dict["gameDuration"] = dict["gameDuration"]
            stat_dict["role"] = x["timeline"]["role"]
            stat_dict["lane"] = x["timeline"]["lane"]
            
            team_pos = 0
            if x["teamId"] == 200:
                team_pos = 1
            
            stat_dict["teamfirstTower"] = dict["teams"][team_pos]["firstTower"]
            stat_dict["teamfirstDragon"] = dict["teams"][team_pos]["firstDragon"]
            stat_dict["teamfirstBaron"] = dict["teams"][team_pos]["firstBaron"]
            
            return stat_dict

def get_champ_dict():
    data = open("/kaggle/input/leaguedata/Champion Ids.txt", "r", encoding="utf-8").read()
    data = json.loads(data)
    champions = [x for x in data["data"]]
    dict = {int(data["data"][x]["key"]): str(x) for x in champions}
    return dict

data = read_text_file()
list_of_df = []

#Create initial DF for analysis
for game in data:
    if "status" not in game:
        id = return_participant_ID(game, name)
        stats = get_stats(game, id)        
        list_of_df.append(pd.DataFrame(stats, index=[0]))

final_df = pd.concat(list_of_df).reset_index(drop=True)

#change champion ID's to champion names
final_df["championId"] = final_df["championId"].apply(lambda x: get_champ_dict()[x].strip())
final_df.rename(columns={"championId": "Champion"}, inplace=True)
print("Number of Double Kills:  {}".format(final_df["doubleKills"].sum()))
print("Number of Triple Kills:  {}".format(final_df["tripleKills"].sum()))
print("Number of Quadra Kills:  {}".format(final_df["quadraKills"].sum()))
print("Number of Penta Kills:   {}".format(final_df["pentaKills"].sum()))
champ_list = final_df["Champion"].value_counts().iloc[0:10].index.to_list()
champ_df = final_df[final_df["Champion"].isin(champ_list)]

sns.set(style="darkgrid")
ax = sns.countplot(x="Champion", data=champ_df, order = champ_list)
ax.set_xticklabels(labels = champ_list, rotation=50)
ax.set_title("Tamoor's 10 Most Played Champions")
plt.show()
labels = ["Win", "Loss"]
counts = [len(final_df[final_df["win"]==True]),len(final_df[final_df["win"]==False])]

fig1, ax1 = plt.subplots()
ax1.pie(counts, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
ax1.set_title("Win/Loss Percentage")

plt.show()
jungle_df = final_df[final_df["lane"]=="JUNGLE"]

# jungle_df["teamfirstDragon"].value_counts()
ax = sns.countplot(x="teamfirstDragon", data=jungle_df)
ax.set(xlabel = "Tamoor's team gets the first dragon")
ax.set_title("Tamoor's Team Obtained the First Dragon")
plt.show()
ax = sns.scatterplot(x="gameDuration", y="totalDamageDealtToChampions", data=jungle_df)
ax.set_title("Champion Damage vs. Game Duration")
ax.set(xlabel = "Game Duration (in minutes)", ylabel="Total Damage to Champs")
ax.set_xticklabels([round(x/60, 0) for x in ax.get_xticks()])
plt.show()
jungle_df.loc[(jungle_df["totalDamageDealtToChampions"] < 8000) & (jungle_df["gameDuration"] > 1920), "Champion"].to_string(index=False)