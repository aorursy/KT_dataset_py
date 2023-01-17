import numpy as np

import pandas as pd

import pickle

import matplotlib.pyplot as plt

import numpy as np

import json

import re 

import time

from pandas.io.json import json_normalize

plt.style.use('ggplot') #ggplot스타일 사용

plt.rc('axes', unicode_minus=False)
chall_game = pd.read_csv('../input/tft-match-data/TFT_Challenger_MatchData.csv')

gr_game = pd.read_csv('../input/tft-match-data/TFT_GrandMaster_MatchData.csv')

master_game = pd.read_csv('../input/tft-match-data/TFT_Master_MatchData.csv')

champ = pd.read_csv('../input/league-of-legends-tftteamfight-tacticschampion/TFT_Champion_CurrentVersion.csv')
# Challenger

data_cons = chall_game.groupby('gameId')['Ranked'].count().tolist()

err_game = []



for i in range(len(data_cons)):

    if data_cons[i] != 8:

        print(chall_game.groupby('gameId')['Ranked'].count().keys()[i])

        err_game.append(chall_game.groupby('gameId')['Ranked'].count().keys()[i])

        



# GrandMaster

data_cons = gr_game.groupby('gameId')['Ranked'].count().tolist()

err_game2 = []



for i in range(len(data_cons)):

    if data_cons[i] != 8:

        print(gr_game.groupby('gameId')['Ranked'].count().keys()[i])

        err_game2.append(gr_game.groupby('gameId')['Ranked'].count().keys()[i])

        

# Master

data_cons = master_game.groupby('gameId')['Ranked'].count().tolist()

err_game3 = []



for i in range(len(data_cons)):

    if data_cons[i] != 8:

        print(master_game.groupby('gameId')['Ranked'].count().keys()[i])

        err_game3.append(master_game.groupby('gameId')['Ranked'].count().keys()[i])

        

# Delete data that does not match the consistency



chall_game = chall_game[chall_game['gameId'] != err_game[0]]

master_game = master_game[master_game['gameId'] != err_game3[0]]
type_df = pd.DataFrame()

type_df['type'] = champ['origin'].unique().tolist()



type_df['1st'] = [2,3,2,3,3,3,3,2,3,2]

type_df['2nd'] = [4,6,4,6,np.nan,6,6,np.nan,np.nan,4]

type_df['3rd'] = [np.nan,np.nan,6,9,np.nan,np.nan,9,np.nan,np.nan,6]





#-----------------

#Suited Match Game data combination names

type_df['type'].iloc[4] = 'MechPilot'

type_df['type'].iloc[0] = 'SpacePirate'

type_df['type'].iloc[8] = 'Set3_Void'

type_df['type'].iloc[2] = 'Set3_Celestial'

type_df['type'].iloc[1] = 'StarGuardian'

type_df['type'].iloc[3] = 'DarkStar'





work = []

for i in champ['class'].tolist():

    work0 = i.split(',')

    for j in work0:

        work.append(''.join(re.findall('[a-zA-Z]',j)))

        

work_df = pd.DataFrame()



#------------

#for unique job name

work_df['work'] = pd.Series(list(set(work))).sort_values()[:4].tolist() + pd.Series(list(set(work))).sort_values()[5:].tolist()



work_df['1st'] = [3,2,2,2,2,2,1,2,2,2,2,1,2]

work_df['2nd'] = [6,4,4,np.nan,4,np.nan,np.nan,4,4,np.nan,4,np.nan,4]

work_df['3rd'] = [9,np.nan,np.nan,np.nan,6,np.nan,np.nan,np.nan,6,np.nan,6,np.nan,np.nan]

work_df['4th'] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,8,np.nan,np.nan]



work_df['work'].iloc[10] = 'Set3_Sorcerer'

work_df['work'].iloc[0] = 'Set3_Blademaster'

work_df['work'].iloc[7] = 'Set3_Mystic'

work_df['work'].iloc[2] = 'Set3_Brawler'



type_dict = {'Space Pirate' : 'SpacePirate',

             'Star Guardian' : 'StarGuardian', 

             'Celestial' : 'StarGuardian',

             'Dark Star' : 'DarkStar',

             'Mech-Pilot' : 'MechPilot',

             'Cybernetic' : 'Cybernetic',

             'Rebel' : 'Rebel',

             'Valkyrie' : 'Valkyrie',

             'Void' : 'Set3_Void',

             'Chrono' : 'Chrono'}



work_dict = {'Blademaster' : 'Set3_Blademaster',

             'Blaster' : 'Blaster',

             'Brawler' : 'Brawler',

             'Demolitionist':'Demolitionist',

             'Infiltrator' : 'Infiltrator',

             'Infiltrato' : 'Infiltrator',

             'ManaReaver' : 'ManaReaver',

             'Mercenary' : 'Mercenary',

             'Mystic' : 'Set3_Mystic',

             'Protector' : 'Protector',

             'Sniper' : 'Sniper',

             'Sorcerer' : 'Set3_Sorcerer',

             'Starship' : 'Starship',

             'Vanguard' : 'Vanguard'}
class_ls = []

for i in range(len(champ)):

    

    class_sr = pd.Series(re.sub('[^,a-zA-Z0-9]','',champ['class'].iloc[i]).split(','))

    

    class_ls.append(class_sr.map(work_dict).tolist())

    

champ['origin'] = champ['origin'].map(type_dict)

champ['class'] = class_ls
# Function



# 분포도 Data Cleansing



def distribution_cleansing(df, rank, specific_rank = False):

    

    if specific_rank == False:

        star1 = []

        star2 = []

        star3 = []

        for i in range(len(df)):



            char = df['champion'].iloc[i]

            char2 = char.replace("'","\"")

            key_ls = list(json.loads(char2).keys())

            value_ls = list(json.loads(char2).values())



            for j in range(len(key_ls)):



                if value_ls[j]['star'] == 1:



                    star1.append(key_ls[j])



                elif value_ls[j]['star'] == 2:



                    star2.append(key_ls[j])



                elif value_ls[j]['star'] == 3:



                    star3.append(key_ls[j])

                    

    elif specific_rank == True:

        df2 = df[df['Ranked']==rank]

        

        star1 = []

        star2 = []

        star3 = []

        for i in range(len(df2)):



            char = df2['champion'].iloc[i]

            char2 = char.replace("'","\"")

            key_ls = list(json.loads(char2).keys())

            value_ls = list(json.loads(char2).values())



            for j in range(len(key_ls)):



                if value_ls[j]['star'] == 1:



                    star1.append(key_ls[j])



                elif value_ls[j]['star'] == 2:



                    star2.append(key_ls[j])



                elif value_ls[j]['star'] == 3:



                    star3.append(key_ls[j])

    

    return star1, star2, star3





def champion_distribution_plot(star1, star2, star3):



    fig,axes = plt.subplots(3,1,figsize = (12,48))



    plot_key0 = pd.Series(star1).value_counts().keys().tolist()

    plot_value0 = pd.Series(star1).value_counts().values.tolist()



    plot_key1 = pd.Series(star2).value_counts().keys().tolist()

    plot_value1 = pd.Series(star2).value_counts().values.tolist()



    plot_key2 = pd.Series(star3).value_counts().keys().tolist()

    plot_value2 = pd.Series(star3).value_counts().values.tolist()







    for idx,ls,ax in zip(['star1','star2','star3'],[[plot_key0,plot_value0],[plot_key1,plot_value1],[plot_key2,plot_value2]],axes):



        ax.barh(ls[0],ls[1])



        #ax.set_title('{} 이웃'.format(n_neighbors))

        ax.set_xlabel('Value')

        ax.set_ylabel('Champion')

        ax.set_title(idx + ' Champion Distribution')
star1_chall, star2_chall, star3_chall = distribution_cleansing(chall_game,'anything')
star1_gma, star2_gma, star3_gma = distribution_cleansing(gr_game,'anything')
star1_ma, star2_ma, star3_ma = distribution_cleansing(master_game,'anything')
champion_distribution_plot(star1_chall,star2_chall,star3_chall)
champion_distribution_plot(star1_gma,star2_gma,star3_gma)
champion_distribution_plot(star1_ma,star2_ma,star3_ma)
# Rank1, Rank2 differences distribution



rank1_star1, rank1_star2, rank1_star3 = distribution_cleansing(chall_game,1,True)



            

rank2_star1 , rank2_star2 , rank2_star3 = distribution_cleansing(chall_game,2,True)
champion_distribution_plot(rank1_star1,rank1_star2,rank1_star3)
champion_distribution_plot(rank2_star1,rank2_star2,rank2_star3)
# Function



def combination_champion_distribution(df, combination_name):

    global type_df, work_df, champ

    

    

    if len(work_df[work_df['work']==combination_name]['1st'])==0:

        work_ok = False

        pass

    else:

        minimum_combi = work_df[work_df['work']==combination_name]['1st'].iloc[0]

        work_ok = True

    

    if len(type_df[type_df['type']==combination_name]['1st'])==0:

        pass

    else:

        minimum_combi = type_df[type_df['type']==combination_name]['1st'].iloc[0]



    

    star1_work_champion, star2_work_champion, star3_work_champion = [], [], []

    star1_type_champion, star2_type_champion, star3_type_champion = [], [], []

    

    for i in range(len(df)):

        

        if i != 0 and i % 10000 ==0 :

            print(i)

            

        char = df['combination'].iloc[i]

        char2 = char.replace("'","\"")

        key_ls = list(json.loads(char2).keys())

        value_ls = list(json.loads(char2).values())

        

        

        char_champ = df['champion'].iloc[i]

        char2_champ = char_champ.replace("'","\"")

        key_ls_champ = list(json.loads(char2_champ).keys())

        value_ls_champ = list(json.loads(char2_champ).values())

        

        for j in range(len(key_ls)):

            if key_ls[j] == combination_name:

                

                if value_ls[j] >= minimum_combi:

                    

                    for k in range(len(key_ls_champ)):

                        

                        if work_ok == True:

                            work_list = champ[champ['name'] == key_ls_champ[k].lower()]['class'].iloc[0]

                            work_list2 = re.sub('[^,a-zA-Z0-9]','',str(work_list)).split(',')

                            

                            

                            if combination_name in work_list2:

                                

                                if value_ls_champ[k]['star'] == 1:

                                    star1_work_champion.append(key_ls_champ[k])

                                    

                                elif value_ls_champ[k]['star'] == 2:

                                    star2_work_champion.append(key_ls_champ[k])

                                    

                                elif value_ls_champ[k]['star'] == 3:

                                    star3_work_champion.append(key_ls_champ[k])

                                    

                        else:

                            type_list = champ[champ['name'] == key_ls_champ[k].lower()]['origin'].iloc[0]

                            type_list2 = re.sub('[^,a-zA-Z0-9]','',type_list).split(',')

                            

                            

                            if combination_name in type_list2:

                                

                                if value_ls_champ[k]['star'] == 1:

                                    star1_type_champion.append(key_ls_champ[k])

                                    

                                elif value_ls_champ[k]['star'] == 2:

                                    star2_type_champion.append(key_ls_champ[k])

                                    

                                elif value_ls_champ[k]['star'] == 3:

                                    star3_type_champion.append(key_ls_champ[k])

                            

                        

                    

                else:

                    continue

                    

            else:

                continue

    

    if  star1_work_champion == []:

        

        return star1_type_champion,star2_type_champion,star3_type_champion

    

    else:

        return star1_work_champion,star2_work_champion,star3_work_champion

    

    

def combination_champion_distribution_plot(star1, star2, star3,combination):



    fig,axes = plt.subplots(3,1,figsize = (12,48))



    plot_key0 = pd.Series(star1).value_counts().keys().tolist()

    plot_value0 = pd.Series(star1).value_counts().values.tolist()



    plot_key1 = pd.Series(star2).value_counts().keys().tolist()

    plot_value1 = pd.Series(star2).value_counts().values.tolist()



    plot_key2 = pd.Series(star3).value_counts().keys().tolist()

    plot_value2 = pd.Series(star3).value_counts().values.tolist()







    for idx,ls,ax in zip(['star1','star2','star3'],[[plot_key0,plot_value0],[plot_key1,plot_value1],[plot_key2,plot_value2]],axes):



        ax.barh(ls[0],ls[1])



        #ax.set_title('{} 이웃'.format(n_neighbors))

        ax.set_xlabel('Value')

        ax.set_ylabel('Champion')

        ax.set_title(combination+' '+idx + ' Champion Distribution')
type_df
star1_champion, star2_champion, star3_champion = combination_champion_distribution(chall_game,'Rebel')
combination_champion_distribution_plot(star1_champion,star2_champion,star3_champion,'Rebel')
star1_champion, star2_champion, star3_champion = combination_champion_distribution(chall_game,'DarkStar')
combination_champion_distribution_plot(star1_champion,star2_champion,star3_champion,'DarkStar')
work_df
star1_champion, star2_champion, star3_champion = combination_champion_distribution(chall_game,'Protector')
combination_champion_distribution_plot(star1_champion,star2_champion,star3_champion,'Protector')
star1_champion, star2_champion, star3_champion = combination_champion_distribution(chall_game,'Blaster')
combination_champion_distribution_plot(star1_champion,star2_champion,star3_champion,'Blaster')