# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting

import matplotlib

matplotlib.style.use('fivethirtyeight')

import os

import seaborn as sns ## plotting histograms



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

tennis_df = pd.read_csv('../input/Data.csv', encoding = 'latin1') # Read Data
tennis_df.WRank = pd.to_numeric(tennis_df.WRank, errors = 'coerce') 

tennis_df.LRank = pd.to_numeric(tennis_df.LRank, errors = 'coerce')

# New Feature: Rank difference between the 2 opponents

tennis_df['Diff'] =  tennis_df.LRank - tennis_df.WRank 

# New Feature: Round the rank difference to 10's and 20's

tennis_df['Round_10'] = 10*round(np.true_divide(tennis_df.Diff,10))

tennis_df['Round_20'] = 20*round(np.true_divide(tennis_df.Diff,20))

# New Feature: Total number of sets in the match

tennis_df['Total Sets'] = tennis_df.Wsets + tennis_df.Lsets



tennis_df.W3 = tennis_df.W3.fillna(0)

tennis_df.W4 = tennis_df.W4.fillna(0)

tennis_df.W5 = tennis_df.W5.fillna(0)

tennis_df.L3 = tennis_df.L3.fillna(0)

tennis_df.L4 = tennis_df.L4.fillna(0)

tennis_df.L5 = tennis_df.L5.fillna(0)



tennis_df['Sets Diff'] = tennis_df.W1+tennis_df.W2+tennis_df.W3+tennis_df.W4+tennis_df.W5 - (tennis_df.L1+tennis_df.L2+tennis_df.L3+tennis_df.L4+tennis_df.L5)

new_df = tennis_df



# 2 New Data Frames: Grand Slam data frame (GS) and non-Grand Slam data frame (non GS)

df_non_GS = new_df[~(new_df.Series == 'Grand Slam')]

df_GS = new_df[new_df.Series == 'Grand Slam']
#%% Winning probability vs Rank Difference

plt.figure(figsize = (10,10))

bins = np.arange(10,200,10)

Gs_prob = []

non_Gs_prob = []



for value in bins:

    pos = value

    neg = -value

    

    pos_wins = len(df_GS[df_GS.Round_10 == pos])

    neg_wins = len(df_GS[df_GS.Round_10 == neg])

    Gs_prob.append(np.true_divide(pos_wins,pos_wins + neg_wins))

    

    pos_wins = len(df_non_GS[df_non_GS.Round_10 == pos])

    neg_wins = len(df_non_GS[df_non_GS.Round_10 == neg])

    non_Gs_prob.append(np.true_divide(pos_wins,pos_wins + neg_wins))

    

    

plt.bar(bins,Gs_prob,width = 9, color = 'red') 

plt.bar(bins,non_Gs_prob,width = 8, color = 'blue')

plt.title('Winning probability vs Rank difference', fontsize = 30)

plt.xlabel('Rank Difference',fontsize = 15)

plt.ylabel('Winning Probability',fontsize = 15)

plt.xlim([10,200])

plt.ylim([0.5,0.9])

plt.legend(['Grand Slams', 'Non Grand Slams'], loc = 1, fontsize = 15)

plt.show()   
plt.figure(figsize = (10,10))

bins = np.arange(10,200,10)



temp_df = df_GS

prob_gs = []



for rank_diff in bins:



    pos = rank_diff

    neg = -rank_diff

    rank_diff_df_pos = temp_df[temp_df.Round_10 == pos]

    w1 = np.sum(rank_diff_df_pos.Wsets)

    l1 = np.sum(rank_diff_df_pos.Lsets)

    

    rank_diff_df_neg = temp_df[temp_df.Round_10 == neg]

    l2 = np.sum(rank_diff_df_neg.Wsets)

    w2 = np.sum(rank_diff_df_pos.Lsets)

    

    w = w1 + w2

    l = l1 + l2

    prob_gs.append(np.true_divide(w, l + w))

    

temp_df = df_non_GS

prob_non_gs = []



for rank_diff in bins:

    

    pos = rank_diff

    neg = -rank_diff

    rank_diff_df_pos = temp_df[temp_df.Round_10 == pos]

    w1 = np.sum(rank_diff_df_pos.Wsets)

    l1 = np.sum(rank_diff_df_pos.Lsets)

    

    rank_diff_df_neg = temp_df[temp_df.Round_10 == neg]

    l2 = np.sum(rank_diff_df_neg.Wsets)

    w2 = np.sum(rank_diff_df_pos.Lsets)

    

    w = w1 + w2

    l = l1 + l2

    prob_non_gs.append(np.true_divide(w, l + w))

    



#plt.hold(True)

plt.bar(bins,prob_gs,  width = 9, color = 'red',)

plt.bar(bins,prob_non_gs,  width = 8, color = 'blue')

plt.title('Winning Probability vs Rank Difference: Single Set')

plt.legend(['Grand Slam', 'Non Grand Slam'], loc = 2, fontsize = 20)

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability per set')

plt.ylim([0.5,0.9])

plt.xlim([10,200])

plt.show()
def Set_prob(pos_diff,set_num,df,round_factor):

       

    diff_df = df[df[round_factor] == pos_diff]

    diff_df = diff_df[diff_df['Total Sets']>=set_num]

    feat1 = "W" + str(set_num)

    feat2 = "L" + str(set_num)

    set_df = diff_df[diff_df[feat1]>diff_df[feat2]]

    w1 = len(set_df)

    l1 = len(diff_df) - w1

            

    diff_df = df[df[round_factor] == -pos_diff]

    diff_df = diff_df[diff_df['Total Sets']>=set_num]

    feat1 = "W" + str(set_num)

    feat2 = "L" + str(set_num)

    set_df = diff_df[diff_df[feat1]>diff_df[feat2]]

    l2 = len(set_df)

    w2 = len(diff_df) - l2

            

    w = w1 + w2

    l = l1 + l2

            

    return np.true_divide(w,l+w)

   

                

bins = np.arange(20,140,20)

prob_1 = []

prob_2 = []

prob_3 = []





for rank_diff in bins:

    

    prob_1.append(Set_prob(rank_diff,1,df_non_GS,"Round_20"))

    prob_2.append(Set_prob(rank_diff,2,df_non_GS,"Round_20"))

    prob_3.append(Set_prob(rank_diff,3,df_non_GS,"Round_20"))

    

plt.figure(figsize = (10,10))

plt.hold(True)

plt.plot(bins,prob_1)

plt.plot(bins,prob_2)

plt.plot(bins,prob_3) 

plt.ylim([0.5,0.9])

plt.legend(['Set 1', 'Set 2', 'Set 3', 'Set 4','Set 5'], loc = 2, fontsize = 20)  

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')

plt.title('Non-Grand Slam Matches')
bins = np.arange(20,140,20)

prob_1 = []

prob_2 = []

prob_3 = []

prob_4 = []

prob_5 = []



for rank_diff in bins:

    

    prob_1.append(Set_prob(rank_diff,1,df_GS,"Round_20"))

    prob_2.append(Set_prob(rank_diff,2,df_GS,"Round_20"))

    prob_3.append(Set_prob(rank_diff,3,df_GS,"Round_20"))

    prob_4.append(Set_prob(rank_diff,4,df_GS,"Round_20"))

    prob_5.append(Set_prob(rank_diff,5,df_GS,"Round_20"))

    

plt.figure(figsize = (10,10))

plt.hold(True)

plt.plot(bins,prob_1)

plt.plot(bins,prob_2)

plt.plot(bins,prob_3) 

plt.plot(bins,prob_4)

plt.plot(bins,prob_5) 

plt.ylim([0.5,0.9])

plt.legend(['Set 1', 'Set 2', 'Set 3', 'Set 4','Set 5'], loc = 2, fontsize = 20)  

plt.xlabel('Rank Difference')

plt.ylabel('Winning Probability')

plt.title('Grand Slam Matches')
# Build the Players Data Frame



#Append the unique values of the winners and losers columns

winners = np.unique(new_df.Winner)

losers = np.unique(new_df.Loser)

players = np.append(winners,losers)

players_un = np.unique(players)

record = np.zeros(len(players_un)) # General record of the player

GS_record = np.zeros(len(players_un)) # Grand Slam record

Clay_record =  np.zeros(len(players_un)) # Clay Record

Carpet_record = np.zeros(len(players_un)) # Carpet Record

Grass_record = np.zeros(len(players_un)) # Grass Record

Hard_record = np.zeros(len(players_un)) #Hard surface record

fifth_set_record = np.zeros(len(players_un)) # Fifth Set record 

the_final_record = np.zeros(len(players_un)) # Fianls Record



d = {'Player_name': players_un, 'record':record, 'GS_record': GS_record,'Clay_record': Clay_record, 'Carpet_record': Carpet_record,'Grass_record':Grass_record,'Hard_record':Hard_record,'fifth_set_recrod':fifth_set_record,'the_final_record':the_final_record }

players_df = pd.DataFrame(data=d)



# Fill in features values for each feature

for i,row in enumerate(players_df.iterrows()):

    w = len(new_df[new_df.Winner == row[1].Player_name])

    l = len(new_df[new_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_Games'] = w + l

    players_df.loc[i,'record'] = np.true_divide(w,(w+l))

    

    temp_df = new_df[new_df.Series == 'Grand Slam']

    w = len(temp_df[temp_df.Winner == row[1].Player_name])

    l = len(temp_df[temp_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_GS_Games'] = w + l

    players_df.loc[i,'GS_record'] = np.true_divide(w,(w+l))

    

    temp_df = new_df[new_df.Surface == 'Clay']

    w = len(temp_df[temp_df.Winner == row[1].Player_name])

    l = len(temp_df[temp_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_Clay_Games'] = w + l

    players_df.loc[i,'Clay_record'] = np.true_divide(w,(w+l))

    

    temp_df = new_df[new_df.Surface == 'Carpet']

    w = len(temp_df[temp_df.Winner == row[1].Player_name])

    l = len(temp_df[temp_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_Carpet_Games'] = w + l

    players_df.loc[i,'Carpet_record'] = np.true_divide(w,(w+l))

    

    temp_df = new_df[new_df.Surface == 'Grass']

    w = len(temp_df[temp_df.Winner == row[1].Player_name])

    l = len(temp_df[temp_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_Grass_Games'] = w + l

    players_df.loc[i,'Grass_record'] = np.true_divide(w,(w+l))

    

    temp_df = new_df[new_df.Surface == 'Hard']

    w = len(temp_df[temp_df.Winner == row[1].Player_name])

    l = len(temp_df[temp_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_Hard_Games'] = w + l

    players_df.loc[i,'Hard_record'] = np.true_divide(w,(w+l))

    

    temp_df = new_df[new_df['Total Sets'] == 5]

    w = len(temp_df[temp_df.Winner == row[1].Player_name])

    l = len(temp_df[temp_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_fifth_set_recrod_Games'] = w + l

    players_df.loc[i,'fifth_set_recrod'] = np.true_divide(w,(w+l))

    

    temp_df = new_df[new_df['Round'] == 'The Final']

    w = len(temp_df[temp_df.Winner == row[1].Player_name])

    l = len(temp_df[temp_df.Loser == row[1].Player_name])

    players_df.loc[i,'Total_final_Games'] = w + l

    players_df.loc[i,'the_final_recrod'] = np.true_divide(w,(w+l))



#GS Factor represents how well a play performs in Grand Slams compared to regular tournaments

players_df['GS_Factor'] = (players_df.GS_record - players_df.record)/(players_df.GS_record + players_df.record)

# Final factor represents how well a play performs in finals compared to regular matches

players_df['Final_Factor'] = (players_df.the_final_recrod - players_df.record)/(players_df.the_final_recrod + players_df.record)

serious_players_df = players_df[players_df.Total_GS_Games > 10]

serious_players_df = serious_players_df[serious_players_df.Total_final_Games>10]
from sklearn.cluster import KMeans

# Semi Serious Players are players who played enough matches on all three major surfaces

semi_serious_players_df = players_df[players_df.Total_Games>163]

data= {'Clay_record': semi_serious_players_df.Clay_record,'Grass_record' :semi_serious_players_df.Grass_record,'Hard_record':semi_serious_players_df.Hard_record} #'fifth_set_recrod':fifth_set_record,'the_final_record':the_final_record }

kmeans_df =  pd.DataFrame(data=data)



kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)

semi_serious_players_df['label'] = kmeans.labels_
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize = (12,12))

ax = fig.add_subplot(111, projection='3d')

x1 = np.array(semi_serious_players_df.Clay_record[semi_serious_players_df.label == 0])

y1 = np.array(semi_serious_players_df.Grass_record[semi_serious_players_df.label == 0])

z1 = np.array(semi_serious_players_df.Hard_record[semi_serious_players_df.label == 0])



x2 = np.array(semi_serious_players_df.Clay_record[semi_serious_players_df.label == 1])

y2 = np.array(semi_serious_players_df.Grass_record[semi_serious_players_df.label == 1])

z2 = np.array(semi_serious_players_df.Hard_record[semi_serious_players_df.label == 1])



x3 = np.array(semi_serious_players_df.Clay_record[semi_serious_players_df.label == 2])

y3 = np.array(semi_serious_players_df.Grass_record[semi_serious_players_df.label == 2])

z3 = np.array(semi_serious_players_df.Hard_record[semi_serious_players_df.label == 2])



plt.hold(True)

ax.scatter(x1,y1, zs = z1, zdir='z', s=70, c= 'r',depthshade=True)

ax.scatter(x2,y2, zs = z2, zdir='z', s=70, c= 'g',depthshade=True)

ax.scatter(x3,y3, zs = z3, zdir='z', s=70, c= 'b',depthshade=True)

ax.set_xlabel('Clay record', fontsize = 20)

ax.set_ylabel('Grass record', fontsize = 20)

ax.set_zlabel('Hard record', fontsize = 20)

ax.set_xlim([0,1])

ax.set_ylim([0,1])

ax.set_zlim([0,1])



plt.legend(['Clay Players', 'Grass Players', 'Good Players'], loc = 2, fontsize = 20)
plt.figure(figsize=(10,10))

plt.hold(True)

plt.plot(x1,y1,'ro')

plt.plot(x2,y2,'go')

plt.plot(x3,y3,'bo')

plt.xlabel('Clay record',fontsize = 20)

plt.ylabel('Grass record',fontsize = 20)

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(['Clay Players', 'Grass Players', 'Good Players'], loc = 2, fontsize = 20)



plt.figure(figsize=(10,10))

plt.hold(True)

plt.plot(y1,z1,'ro')

plt.plot(y2,z2,'go')

plt.plot(y3,z3,'bo')

plt.xlabel('Grass record', fontsize = 20)

plt.ylabel('Hard record',fontsize = 20)

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(['Clay Players', 'Grass Players', 'Good Players'], loc = 2, fontsize = 20)



plt.figure(figsize=(10,10))

plt.hold(True)

plt.plot(z1,x1,'ro')

plt.plot(z2,x2,'go')

plt.plot(z3,x3,'bo')

plt.xlabel('Hard record',fontsize = 20)

plt.ylabel('Clay record',fontsize = 20)

plt.xlim([0,1])

plt.ylim([0,1])

plt.legend(['Clay Players', 'Grass Players', 'Good Players'], loc = 2, fontsize = 20)
federer = np.zeros(3)

federer[0] = semi_serious_players_df.Clay_record[semi_serious_players_df.Player_name == 'Federer R.']

federer[1] = semi_serious_players_df.Grass_record[semi_serious_players_df.Player_name == 'Federer R.']

federer[2] = semi_serious_players_df.Hard_record[semi_serious_players_df.Player_name == 'Federer R.']



nadal = np.zeros(3)

nadal[0] = semi_serious_players_df.Clay_record[semi_serious_players_df.Player_name == 'Nadal R.']

nadal[1] = semi_serious_players_df.Grass_record[semi_serious_players_df.Player_name == 'Nadal R.']

nadal[2] = semi_serious_players_df.Hard_record[semi_serious_players_df.Player_name == 'Nadal R.']



djokovich = np.zeros(3)

djokovich[0] = semi_serious_players_df.Clay_record[semi_serious_players_df.Player_name == 'Djokovic N.']

djokovich[1] = semi_serious_players_df.Grass_record[semi_serious_players_df.Player_name == 'Djokovic N.']

djokovich[2] = semi_serious_players_df.Hard_record[semi_serious_players_df.Player_name == 'Djokovic N.']



plt.figure(figsize = (10,10))

plt.hold(True)

plt.bar([0,2,4],federer, color = 'green' , width = 0.5)

plt.bar([0.5,2.5,4.5],nadal, color = 'red', width = 0.5)

plt.bar([1,3,5],djokovich, color = 'blue', width = 0.5)

plt.xlim(0,5.5)

plt.xticks([0.5,2.5,4.5], ['Clay','Grass','Hard'], fontsize = 25)

plt.legend(['Federer','Nadal','Djokovic'], fontsize = 15)

plt.ylabel('Winning Probability')

plt.show()