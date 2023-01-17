# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/games.csv")
data.head()
data = data.loc[:,"gameDuration" :]

data # now our new data is created
data.winner = ["Blue"  if i == 1 else "Red" for i in data["winner"]]





data.gameDuration = [float(sec/60) for sec in data.gameDuration]

data.rename(columns = {"gameDuration" : "gameDuration_minutes"}, inplace = True)

 



data
# We can find some useful datas inside of our dataset

red_count = 0

blue_count = 0



for i in data.winner:

    if "Red" in i:

        red_count += 1

    elif "Blue" in i:

        blue_count += 1

        

print("Blue team wins {} times  and Red team wins {} times.".format(blue_count, red_count))



summ =0

count =0

for i in data.gameDuration_minutes:

    summ = summ +i

    count += 1

average = float(summ) /float(count)



print(" The average game length is {}".format(average))



data.describe()



    

    
filter1 = data.gameDuration_minutes > 50



new_data = data[filter1]



new_data = new_data.loc[:,["winner","gameDuration_minutes","t1_towerKills","t1_inhibitorKills","t1_baronKills","t1_dragonKills","t2_towerKills",

            "t2_inhibitorKills","t2_baronKills","t2_dragonKills"]]





inhip_total = new_data.t1_inhibitorKills + new_data.t2_inhibitorKills

new_data["Total_inhibitor"] = inhip_total

tower_total = new_data.t1_towerKills + new_data.t2_towerKills

new_data["Total tower"] = tower_total

baron_total = new_data.t1_baronKills + new_data.t2_baronKills

new_data["Total baron"] = baron_total

dragon_total = new_data.t1_dragonKills + new_data.t2_dragonKills

new_data["Total dragon"] = dragon_total



new_data
import plotly.graph_objs as go

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import matplotlib.pyplot as plt





trace1 = go.Scatter(

                    x = new_data["gameDuration_minutes"],

                    y = new_data["Total dragon"],

                    mode = "lines",

                    name = "dragons",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= new_data.winner)



trace2 = go.Scatter(

                    x = new_data["gameDuration_minutes"],

                    y = new_data["Total baron"],

                    mode = "lines",

                    name = "barons",

                    marker = dict(color = 'rgba(150, 5, 33, 0.8)'),

                    text= new_data.winner)





data = [trace1, trace2]



layout = dict(title = 'Baron vs Dragon',

              xaxis= dict(title= 'Minutes',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)

import plotly.graph_objs as go

# creating trace1

trace1 =go.Scatter(

                    x = new_data.t1_inhibitorKills,

                    y = new_data.gameDuration_minutes,

                    mode = "markers",

                    name = "Blue inhibitor numbers",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= new_data.winner)

# creating trace2

trace2 =go.Scatter(

                    x = new_data.t2_inhibitorKills,

                    y = new_data.gameDuration_minutes,

                    mode = "markers",

                    name = "Red inhibitor numbers",

                    marker = dict(color = 'rgba(255, 128, 2, 0.8)'),

                    text= new_data.winner)



data = [trace1, trace2]

layout = dict(title = 'Inhibitor numbers respect to minutes',

              xaxis= dict(title= 'Inhibitors',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Minutes',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
color_list = ["blue" if i== "Blue" else "red" for i in new_data["winner"]]



x = new_data["gameDuration_minutes"]

y = new_data["Total_inhibitor"]

plt.scatter(x,y, c = color_list, alpha = 0.5)

plt.title("Red and Blue team comparison")

plt.xlabel("Minutes")

plt.ylabel("Total inhibitor")

plt.show()

new_data.winner = [1 if i == "Blue" else 0 for i in new_data.winner]







x_data = new_data.drop(["winner"], axis = 1)



x = (x_data - np.min(x_data))/(np.max(x_data)- np.min(x_data)) # normalization



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)





from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=3) # k value is 3



knn.fit(x_train, y_train)





print("{}nn score : {}".format(3,knn.score(x_test,y_test)))



champ_data = pd.read_json("../input/champion_info_2.json")

champions = pd.read_json((champ_data["data"]).to_json(), orient="index")

champions.set_index(["id"], inplace = True)

champions



champList = ["t1_champ1id","t1_champ2id","t1_champ3id","t1_champ4id","t1_champ5id",

             "t2_champ1id","t2_champ2id","t2_champ3id","t2_champ4id","t2_champ5id"]



data = pd.read_csv("../input/games.csv")





def conversion(x):

    champ = champions["name"][x]

    return champ





for column in champList:

    data[column] = data[column].apply(lambda x: conversion(x))

    



banList = ["t1_ban1","t1_ban2", "t1_ban3", "t1_ban4", "t1_ban5",

           "t2_ban1", "t2_ban2", "t2_ban4", "t2_ban5"]



for column in banList:

    data[column] = data[column].apply(lambda x : conversion(x))

    





summ_data = pd.read_json("../input/summoner_spell_info.json")

summoners = pd.read_json((summ_data["data"]).to_json(), orient="index")



def summ_conversion(x):

    summoner = summoners["name"][x]

    return summoner



summList = ["t1_champ1_sum1","t1_champ1_sum2","t1_champ2_sum1","t1_champ2_sum2","t1_champ3_sum1",

                 "t1_champ3_sum2","t1_champ4_sum1","t1_champ4_sum2","t1_champ5_sum1","t1_champ5_sum2",

                 "t2_champ1_sum1","t2_champ1_sum2","t2_champ2_sum1","t2_champ2_sum2","t2_champ3_sum1",

                 "t2_champ3_sum2","t2_champ4_sum1","t2_champ4_sum2",

                 "t2_champ5_sum1","t2_champ5_sum2"]





for column in summList:

    data[column] = data[column].apply(lambda x : summ_conversion(x))





data



picksData = pd.concat([data.t1_champ1id, data.t1_champ2id, data.t1_champ3id, data.t1_champ4id,

                      data.t1_champ5id, data.t2_champ1id, data.t2_champ2id, data.t2_champ3id,

                      data.t2_champ4id, data.t2_champ5id], axis = 0, ignore_index=True)







winner_list =[]

a = 0

count =0

while a < 10 :

    for each in data.winner:

        if each == 1:

            winner_list.append(1)

        else:

            winner_list.append(2)

        count +=1

    a += 1





winner_data = pd.DataFrame(winner_list)



new_data = pd.concat([winner_data, picksData], axis = 1)



#picksData



#new_data

#new_data



new_data.iloc[:,1]