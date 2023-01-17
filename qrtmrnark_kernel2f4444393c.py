# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import json



#RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

#Confusion Matrix

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix, plot_confusion_matrix

#XGBOOST

from xgboost import XGBRegressor



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



from sklearn.feature_extraction.text import CountVectorizer

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

import re





from numpy import array

from numpy import argmax

from keras.layers.core import Dense, Activation, Dropout

from keras.preprocessing import sequence





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
    PATH = '/kaggle/input/csgoai/dataset_initial/'

    subpath = PATH + 'dataset_00.json'

    first_line = []

    with open(subpath, 'r') as f: 

        first_line.append(f.readline())

    f.close()



    jdata = json.loads(first_line[0])

    df = pd.DataFrame(jdata)



    for i, file in enumerate(os.listdir(PATH)): 

        subpath = PATH + file

        first_line = []

        print("Loading file: {}/{}".format(i+1,len(os.listdir(PATH))))

        with open(subpath, 'r') as f: 

            first_line.append(f.readline())

            f.close()

            jdata = json.loads(first_line[0])

            df = df.append(jdata, ignore_index = True)

        if(i == 24): 

            break

    #df.to_csv('mycsvfile.csv',index=False)



#df = pd.read_csv('/kaggle/working/mycsvfile.csv')        
df = df.drop(columns = ['patch_version', 'map_crc'])
def Insert_row(row_number, df_, row_value): 

    # Starting value of upper half 

    start_upper = 0

   

    # End value of upper half 

    end_upper = row_number 

   

    # Start value of lower half 

    start_lower = row_number 

   

    # End value of lower half 

    end_lower = df_.shape[0] 

   

    # Create a list of upper_half index 

    upper_half = [*range(start_upper, end_upper, 1)] 

   

    # Create a list of lower_half index 

    lower_half = [*range(start_lower, end_lower, 1)] 

   

    # Increment the value of lower half by 1 

    lower_half = [x.__add__(1) for x in lower_half] 

   

    # Combine the two lists 

    index_ = upper_half + lower_half 

   

    # Update the index of the dataframe 

    df_.index = index_ 

   

    # Insert a row at the end 

    df_.loc[row_number] = row_value 

    

    # Sort the index labels 

    df_ = df_.sort_index() 

   

    # return the dataframe 

    return df_
df
def get_team_value(x):

    

    weapon_dict = {

        "Deagle": 700,

        "Elite": 400,

        "FiveSeven": 500,

        "Glock": 200,

        "Ak47": 2700,

        "Aug": 3300,

        "Awp": 4750,

        "Famas": 2050,

        "G3sg1": 5000,

        "GalilAR": 1800,

        "M249": 5200,

        "M4a4": 3100,

        "Mac10": 1050,

        "P90": 2350,

        "Mp5sd": 1500,

        "Ump45": 1200,

        "Xm1014": 2000,

        "Bizon": 1400,

        "Mag7": 1300,

        "Negev": 1700,

        "Sawedoff": 1100,

        "Tec9": 500,

        "ZeusX27": 300,

        "P2000": 200,

        "Mp7": 1500,

        "Mp9": 1250,

        "Nova": 1050,

        "P250": 300,

        "Scar20": 5000,

        "Sg553": 3000,

        "Ssg08": 1700,

        "M4a1S": 2900,

        "UspS": 200,

        "Cz75Auto": 500,

        "R8Revolver": 600,

        "Flashbang": 200,

        "HeGrenade": 300,

        "SmokeGrenade": 300,

        "MolotovGrenade": 400,

        "DecoyGrenade": 50,

        "IncendiaryGrenade": 600,

        "Deagle": 700,

        "Elite": 400,

        "FiveSeven": 500,

        "Glock": 200,

        "Ak47": 2700,

        "Aug": 3300,

        "Awp": 4750,

        "Famas": 2050,

        "G3sg1": 5000,

        "GalilAR": 1800,

        "M249": 5200,

        "M4a4": 3100,

        "Mac10": 1050,

        "P90": 2350,

        "Mp5sd": 1500,

        "Ump45": 1200,

        "Xm1014": 2000,

        "Bizon": 1400,

        "Mag7": 1300,

        "Negev": 1700,

        "Sawedoff": 1100,

        "Tec9": 500,

        "ZeusX27": 300,

        "P2000": 200,

        "Mp7": 1500,

        "Mp9": 1250,

        "Nova": 1050,

        "P250": 300,

        "Scar20": 5000,

        "Sg553": 3000,

        "Ssg08": 1700,

        "M4a1S": 2900,

        "UspS": 200,

        "Cz75Auto": 500,

        "R8Revolver": 600,

        "Flashbang": 200,

        "HeGrenade": 300,

        "SmokeGrenade": 300,

        "MolotovGrenade": 400,

        "DecoyGrenade": 50,

        "IncendiaryGrenade": 600

    }

    

    t_value = []

    ct_value = []



    for round_ in x:

        t_v = 0

        ct_v = 0

        t_p = 0

        ct_p = 0

        for player in round_:

            inventory = player['inventory']



            for item in inventory:

                if item['item_type'] in weapon_dict:

                    if (player['team'] == 'CT'):

                        ct_p += 1

                        ct_v += weapon_dict.get(item['item_type'])

                    else:

                        t_p+= 1

                        t_v += weapon_dict.get(item['item_type'])

        if(t_p == 0):

            t_p = 1

        if(ct_p == 0):

            ct_p = 1

            

        if(t_v == 0):

            t_value.append(0)

        else:

            t_value.append(t_v/t_p)

        if(ct_v == 0):

            ct_value.append(0)

        else:

            ct_value.append(ct_v/ct_p)



    print(len(t_value))

    print(len(ct_value))



    return t_value, ct_value
avg_t_value, avg_ct_value = get_team_value(df['alive_players'])
df['avg_t_value'] = avg_t_value

df['avg_ct_value'] = avg_ct_value
df
def get_team_health_and_player(x): 

    ct_total_h = []

    t_total_h = []

    count_ct = []

    count_t = []

    for round_ in x: 

        ct_h = []

        t_h = []

        for player in round_: 

            health = player['health']

            if(player['team'] == 'CT'): 

                ct_h.append(health)

            else: 

                t_h.append(health)

        count_ct.append(len(ct_h))

        count_t.append(len(t_h))

        

        if(len(ct_h) == 0):

            ct_total_h.append(0)

        else:

            ct_total_h.append(np.mean(ct_h))

        if(len(t_h) == 0):

            t_total_h.append(0)

        else:

            t_total_h.append(np.mean(t_h))

            

    return count_ct, count_t, ct_total_h, t_total_h
alive_ct,alive_t,avg_h_ct,avg_h_t = get_team_health_and_player(df['alive_players'])
df['alive_ct_count'] = alive_ct

df['alive_t_count'] = alive_t

df['avg_h_ct'] = avg_h_ct

df['avg_h_t'] = avg_h_t
def get_team_smokes_molo_flashbang_defuse(x): 

    ct_molos = []

    ct_flashbangs = []

    ct_nades = []

    ct_smokes = []

    ct_defusers = []



    t_molos = []

    t_flashbangs = []

    t_nades = []

    t_smokes = []

    

    for round_ in x: 

        ct_m = 0

        ct_f = 0

        ct_n = 0

        ct_s = 0

        ct_d = 0



        t_m = 0

        t_f = 0

        t_n = 0

        t_s = 0



        for player in round_:

            inventory = player['inventory']

            if(player['has_defuser']):

                    ct_d += 1

            for item in inventory:

                if(item['item_type'] == 'IncendiaryGrenade'):

                    if(player['team'] == 'CT'): 

                        ct_m+=1

                    else: 

                        t_m+=1

                if(item['item_type'] == 'MolotovGrenade'):

                    if(player['team'] == 'CT'): 

                        ct_m+=1

                    else: 

                        t_m+=1

                if(item['item_type'] == 'Flashbang'):

                    if(player['team'] == 'CT'): 

                        ct_f+=1

                    else: 

                        t_f+=1

                if(item['item_type'] == 'HeGrenade'):

                    if(player['team'] == 'CT'): 

                        ct_n+=1

                    else: 

                        t_n+=1

                if(item['item_type'] == 'SmokeGrenade'):

                    if(player['team'] == 'CT'): 

                        ct_s+=1

                    else: 

                        t_s+=1

                        

        ct_molos.append(ct_m)

        ct_flashbangs.append(ct_f)

        ct_nades.append(ct_n)

        ct_smokes.append(ct_s)

        ct_defusers.append(ct_d)



        t_molos.append(t_m)

        t_flashbangs.append(t_f)

        t_nades.append(t_n)

        t_smokes.append(t_s)

            

    return ct_molos, ct_flashbangs, ct_nades, ct_smokes, ct_d, t_molos, t_flashbangs, t_nades, t_smokes
ct_molos, ct_flashbangs, ct_nades, ct_smokes, ct_defusers, t_molos, t_flashbangs, t_nades, t_smokes = get_team_smokes_molo_flashbang_defuse(df.alive_players)
df['ct_smokes'] = ct_smokes

df['ct_molos'] = ct_molos

df['ct_flashbangs'] = ct_flashbangs

df['ct_nades'] = ct_nades

df['defusers'] = ct_defusers

df['t_smokes'] = t_smokes

df['t_molos'] = t_molos

df['t_flashbangs'] = t_flashbangs

df['t_nades'] = t_nades
def point_in_cicle(px, py, cx, cy, r):

    return (not(((px - cx)^2 + (py - cy)^2) > (r^2)))
def strategic_smokes_and_molos(map_, chart):

    ct_smokes = []

    t_smokes = []

    ct_molos = []

    t_molos = []

    

    for round_ in map_:

        mappo = round_['map']

        smokes_coords = mappo['smokes']

        molo_coords = mappo['molos']

        

        ct_smoke = 0

        ct_molo = 0

        t_molo = 0

        t_smoke = 0

        

        for smoke in round_['active_smokes']:

            for coord in smoke_coords:

                # (𝑥𝑝−𝑥𝑐)2+(𝑦𝑝−𝑦𝑐)2 < 𝑟2

                if( point_in_circle(smoke['x'], smoke['y'], coord['x'], coord['y'], coord['r'])):

                    if(coord['side'] == t):

                        t_smoke += 1

                    else:

                        ct_smoke += 1

                        

        for molo in round_['active_molotovs']:

            for coord in molo_coords:

                # (𝑥𝑝−𝑥𝑐)2+(𝑦𝑝−𝑦𝑐)2 < 𝑟2

                if( point_in_circle(molo['x'], molo['y'], coord['x'], coord['y'], coord['r'])):

                    if(coord['side'] == t):

                        t_molo += 1

                    else:

                        ct_molo += 1

                       

        ct_smokes.append(ct_smoke)

        t_smokes.append(t_smoke)

        ct_molos.append(ct_molo)

        t_molos.append(t_molo)

        

        

    return ct_smokes, t_smokes, ct_molos, t_molos
df
def get_armor_attribute(x): 

    ct_armor = []

    ct_helmets = []

    t_armor = []

    t_helmets = []

    

    for round_ in x: 

        ct_a = []

        t_a = []

        ct_h = []

        t_h = []

        for player in round_: 

            armor = player['armor']

            has_h = player['has_helmet']

            if(player['team'] == 'CT'): 

                ct_a.append(armor)

                if(has_h):

                    ct_h.append(1)

                else:

                    ct_h.append(0)

            else: 

                t_a.append(armor)

                if(has_h):

                    t_h.append(1)

                else:

                    t_h.append(0)

        

        #Armor

        if(len(ct_a) == 0):

            ct_armor.append(0)

        else:

            ct_armor.append(np.mean(ct_a))

        

        if(len(t_a) == 0):

            t_armor.append(0)

        else:

            t_armor.append(np.mean(t_a))

        #Helmets

        if(len(ct_h) == 0):

            ct_helmets.append(0)

        else:

            ct_helmets.append(np.mean(ct_h))

            

        if(len(t_h) == 0):

            t_helmets.append(0)

        else:

            t_helmets.append(np.mean(t_h))

            

    return ct_armor, ct_helmets, t_armor, t_helmets
ct_armor, ct_helmets, t_armor, t_helmets = get_armor_attribute(df.alive_players)
df['ct_avg_armor'] = ct_armor

df['t_avg_armor'] = t_armor

df['percent_ct_helmets'] = ct_helmets

df['percent_t_helmets'] = t_helmets
def get_money(x, status, timeleft):

    ct_money = []

    t_money = []

    i = 0

    for round_ in x: 

        

        if(status[i] == 'FreezeTime' or (status[i] == 'Normal' and timeleft[i] > 100)):

            ct_m = 0

            t_m = 0



            for player in round_: 



                money = player['money']



                if(player['team'] == 'CT'): 

                    ct_m += money



                else: 

                    t_m += money





        ct_money.append(ct_m)

        t_money.append(t_m)

        

        i+=1

            

    return ct_money, t_money

    
ct_money, t_money = get_money(df['alive_players'], df['round_status'], df['round_status_time_left'])
df['ct_money'] = ct_money

df['t_money'] = t_money
df
def get_score(x):

    ct_score = []

    t_score = []

    for round_ in x:

        ct_score.append(round_[0])

        t_score.append(round_[1])

        

    return ct_score, t_score

    
ct_score, t_score = get_score(df['current_score'])
df['ct_score'] = ct_score

df['t_score'] = t_score
dummies = pd.get_dummies(df[['map', 'round_status']])

df_total = pd.concat([df, dummies], axis= 1)

df = df.iloc[0:0]
prev = 0;

timeouts = []

prev_freeze = False

max_i = len(df_total.index)

for index, row in df_total.iterrows():

    if(index > max_i):

        break

    elif(row['round_status'] == 'FreezeTime' and df_total['round_status'].iloc[index+1] == 'FreezeTime'):

        timeouts.append(index)

df_total.drop(df_total.index[timeouts], inplace=True)
df_total.reset_index(inplace=True, drop=True)
score = df_total['current_score']
def round_counter(score):

    roundcount = []

    count = 0



    last = 0

    for round_ in score:

            

        if(round_[0] + round_[1] > last):

            last += 1

            count+= 1

            

        elif(round_[0] + round_[1] < last):

            last = 0

            count += 1





        roundcount.append(count)

    

            

    return roundcount
round_ = round_counter(score)
df_total['round'] = round_
#Counts grenade for each round

unique_items = []

temp = []

he_count = 0

for round_ in df_total.alive_players:

    for player in round_:

        for item in player['inventory']:

            if (item['item_type'] == 'HeGrenade'):

                    he_count += 1

    unique_items.append(he_count)

    he_count = 0
pd.Series(unique_items).value_counts()
#Creates boxplot

import seaborn as sns

import matplotlib.pyplot as plt

boxp_columns = ['round_status_time_left', 'alive_ct_count', 'alive_t_count', 'avg_h_ct', 'avg_h_t', 'ct_smokes', 'ct_molos', 'ct_flashbangs', 'ct_nades', 'defusers', 't_smokes', 't_molos', 't_flashbangs', 't_nades', 'ct_avg_armor', 't_avg_armor', 'percent_ct_helmets', 'percent_t_helmets', 'ct_money', 't_money', 'ct_score', 't_score']





for column in boxp_columns:

    sns.boxplot(x=df_total[column])

    plt.show()
rel_columns = [ 'round_status_time_left', 'alive_ct_count', 'alive_t_count', 'avg_h_ct', 'avg_h_t', 'ct_smokes', 'ct_molos', 'ct_flashbangs', 'ct_nades', 't_smokes', 't_molos', 't_flashbangs', 't_nades', 'ct_avg_armor', 't_avg_armor', 'percent_ct_helmets', 'percent_t_helmets', 'ct_money', 't_money', 'ct_score', 't_score']

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))

fig.set_dpi(100)

ax = sns.heatmap(df_total[df_total['round_winner']=='CT'][rel_columns].corr(), ax = axes[0], cmap='coolwarm')

ax.set_title('CT')

ax = sns.heatmap(df_total[df_total['round_winner']=='Terrorist'][rel_columns].corr(), ax = axes[1], cmap='coolwarm')

ax.set_title('T')
rel_columns = [ 'round_status_time_left', 'alive_ct_count', 'alive_t_count', 'avg_h_ct', 'avg_h_t', 'ct_smokes', 'ct_molos', 'ct_flashbangs', 'ct_nades', 't_smokes', 't_molos', 't_flashbangs', 't_nades', 'ct_avg_armor', 't_avg_armor', 'percent_ct_helmets', 'percent_t_helmets', 'ct_money', 't_money', 'ct_score', 't_score']

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))

fig.set_dpi(100)

ax = sns.heatmap(df_total[df_total['round_winner']=='CT'][rel_columns].corr(method='kendall'), ax = axes[0], cmap='coolwarm')

ax.set_title('CT')

ax = sns.heatmap(df_total[df_total['round_winner']=='Terrorist'][rel_columns].corr(method='kendall'), ax = axes[1], cmap='coolwarm')

ax.set_title('T')
df_total['total_ct_nades'] = df_total[['ct_smokes', 'ct_molos', 'ct_flashbangs', 'ct_nades']].sum(axis=1)
df_total['total_t_nades'] = df_total[['t_smokes', 't_molos', 't_flashbangs', 't_nades']].sum(axis=1)
rel_columns = [ 'round_status_time_left', 'alive_ct_count', 'alive_t_count', 'avg_h_ct', 'avg_h_t', 'ct_avg_armor', 't_avg_armor', 'ct_money', 't_money', 'ct_score', 't_score', 'total_t_nades', 'total_ct_nades']

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))

fig.set_dpi(100)

ax = sns.heatmap(df_total[df_total['round_winner']=='CT'][rel_columns].corr(), ax = axes[0], cmap='coolwarm')

ax.set_title('CT')

ax = sns.heatmap(df_total[df_total['round_winner']=='Terrorist'][rel_columns].corr(), ax = axes[1], cmap='coolwarm')

ax.set_title('T')
#Time left doens't have the expected impact on the outcome, so we add time left from freezetime to time left, so that the time is more consistent



for index, row in df_total.iterrows():

    if(row['round_status'] == 'FreezeTime'):

        df_total.at[index, 'round_status_time_left'] = row['round_status_time_left'] + df_total['round_status_time_left'].iloc[index + 1]

    
rel_columns = [ 'round_status_time_left', 'alive_ct_count', 'alive_t_count', 'avg_h_ct', 'avg_h_t', 'ct_avg_armor', 't_avg_armor', 'ct_money', 't_money', 'ct_score', 't_score', 'total_t_nades', 'total_ct_nades']

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))

fig.set_dpi(100)

ax = sns.heatmap(df_total[df_total['round_winner']=='CT'][rel_columns].corr(), ax = axes[0], cmap='coolwarm')

ax.set_title('CT')

ax = sns.heatmap(df_total[df_total['round_winner']=='Terrorist'][rel_columns].corr(), ax = axes[1], cmap='coolwarm')

ax.set_title('T')


current_ct_score = []

current_t_score = []

next_score = []



for index, row in df_total.iterrows():

    current_score = row['current_score']

    if(row['round_winner'] == 'CT'):

        next_score.append([1,0])

    

    else:

        next_score.append([0,1])

    

    current_ct_score.append(current_score[0])

    current_t_score.append(current_score[1])

    

df_total['current_ct_score'] = current_ct_score

df_total['current_t_score'] = current_t_score

df_total['next_score'] = next_score
rel_columns.append('round')

rel_columns.append('avg_t_value')

rel_columns.append('avg_ct_value')

drop_list = []

for elm in df_total.columns:

    if(elm not in rel_columns):

        drop_list.append(elm)
x_set = df_total.drop(columns=drop_list).to_numpy()

y_set = df_total['next_score'].to_numpy()
X = df_total.drop(columns=drop_list)

y = df_total['next_score']
train_idx, validate_idx, test_idx = np.split(pd.Series(df_total['round'].unique()).sample(frac=1), [int(.6*len(pd.Series(df_total['round'].unique()))), int(.8*len(pd.Series(df_total['round'].unique())))])
df_total['next_score_bool'] = df_total['round_winner'].apply(lambda x: 1 if x == 'CT' else 0)
highest = 0

round_ = -1

current_count = 0

highest_index = 0

for index, row in df_total.iterrows():

    if(row['round'] > round_):

        current_count = 1

        round_ = row['round']

    else:

        current_count += 1

        if(current_count > highest):

            highest = current_count

            highest_index = index

        
def empty_row(round_boi, df):

    empty_row = []

    for x in range(len(df.columns)):

        if(df.columns[x] == 'round'):

            empty_row.append(round_boi)

        else:

            empty_row.append(0)

    return(empty_row)


round_ = 0

current_count = 0

fill_index_amount_round = []

for index, row in df_total.iterrows():

    if(row['round'] > round_):

        if(current_count < 9):

            fill_index_amount_round.append([index-1, 9-current_count, round_])

        current_count = 1

        round_ = row['round']

        

    elif(index == len(df_total.index)-1):

        if(current_count < 9):

            fill_index_amount_round.append([index, 8-current_count, round_])

        current_count = 1

        round_ = row['round']

    else:

        current_count += 1

        
#df_total_empty_rows = df_total

#offset = 1

#for i in fill_index_amount_round:

#    for j in range(i[1]):

#        df_total_empty_rows = Insert_row(i[0] + offset, df_total_empty_rows, empty_row(i[2], df_total_empty_rows))

#        offset += 1
def standardize(df, label):

    

    #standardizes a series with name ``label'' within the pd.DataFrame ``df''.

    

    df = df.copy(deep=True)

    series = df.loc[:, label]

    avg = series.mean()

    stdv = series.std()

    series_standardized = (series - avg)/ stdv

    return series_standardized
def standardize_df(df):

    for x in df.drop(columns=drop_list).columns:

        if(x != 'round' and x != 'next_score_bool' and x != 'next_score'):

            df[x] = standardize(df, x)

    return df
df_total_normalized = standardize_df(df_total)
#temp = df_total_empty_rows

#df_total_empty_rows_normalized = standardize_df(temp)
drop_list.append('next_score_bool')

drop_list.append('next_score')

temp_list = drop_list

temp_list.append('round')

train_X_test = df_total_normalized[df_total_normalized['round'].isin(train_idx)].drop(columns=temp_list)

train_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(train_idx)]



validate_X_test = df_total_normalized[df_total_normalized['round'].isin(validate_idx)].drop(columns=temp_list)

validate_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(validate_idx)]



test_X_test = df_total_normalized[df_total_normalized['round'].isin(test_idx)].drop(columns=temp_list)

test_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(test_idx)]
def random_forest(train_X, train_y):

    clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=50)

    clf.fit(train_X, train_y)

    %matplotlib inline

    importances = clf.feature_importances_

    std = np.std([tree.feature_importances_ for tree in clf.estimators_],

                 axis=0)

    indices = np.argsort(importances)[::-1]



    # Print the feature ranking

    print("Feature ranking:")



    for f in range(train_X_test.shape[1]):

        print("{} feature {} ({})".format(f + 1, train_X_test.columns[indices[f]], importances[indices[f]]))



    # Plot the impurity-based feature importances of the forest

    plt.figure()

    plt.title("Feature importances")

    plt.bar(range(train_X.shape[1]), importances[indices],

            color="r", yerr=std[indices], align="center")

    plt.xticks(range(train_X.shape[1]), indices)

    plt.xlim([-1, train_X.shape[1]])

    plt.show()
random_forest(train_X_test, train_y_test)
def confusion_mat(train_X, train_y, validate_X, validate_y):



    clf = RandomForestClassifier(max_depth=5, random_state=0, n_estimators=50)

    clf.fit(train_X, train_y)

    pred = clf.predict(validate_X)





    conf_mat = confusion_matrix(validate_y, pred, normalize='true')

    print(conf_mat)



    disp = plot_confusion_matrix(clf, validate_X, validate_y,

    display_labels=['CT', 'T'], cmap=plt.cm.Blues, normalize='true')
confusion_mat(train_X_test, train_y_test, validate_X_test, validate_y_test)
def xgboost_(trainX, trainy, validateX, validatey):

    my_model = XGBRegressor()

    # Add silent=True to avoid printing out updates with each cycle

    my_model.fit(trainX, trainy, verbose=False)

    

    # make predictions

    predictions = my_model.predict(validateX)



    from sklearn.metrics import mean_absolute_error

    print("Mean Absolute Error : " + str(mean_absolute_error(predictions, validatey)))
xgboost_(train_X_test, train_y_test, validate_X_test, validate_y_test)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_X_test, train_y_test, early_stopping_rounds=5, 

             eval_set=[(validate_X_test, validate_y_test)], verbose=False)
# make predictions

predictions = my_model.predict(validate_X_test)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, validate_y_test)))
drop_list.append('next_score_bool')

drop_list.append('next_score')

drop_list.append('round')

drop_list.append('ct_score')

drop_list.append('t_score')



train_X_test = df_total_normalized[df_total_normalized['round'].isin(train_idx)].drop(columns=drop_list)

train_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(train_idx)]



validate_X_test = df_total_normalized[df_total_normalized['round'].isin(validate_idx)].drop(columns=drop_list)

validate_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(validate_idx)]



test_X_test = df_total_normalized[df_total_normalized['round'].isin(test_idx)].drop(columns=drop_list)

test_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(test_idx)]

random_forest(train_X_test, train_y_test)
confusion_mat(train_X_test, train_y_test, validate_X_test, validate_y_test)
xgboost_(train_X_test, train_y_test, validate_X_test, validate_y_test)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_X_test, train_y_test, early_stopping_rounds=5, 

             eval_set=[(validate_X_test, validate_y_test)], verbose=False)
# make predictions

predictions = my_model.predict(validate_X_test)



print("Mean Absolute Error : " + str(mean_absolute_error(predictions, validate_y_test)))
drop_list.append('next_score_bool')

drop_list.append('next_score')

drop_list.append('round')

drop_list.append('ct_score')

drop_list.append('t_score')
def encode_list_Yx9(df_, list_idx):

    idx = 0

    y_set = []

    

    temp_df = df_[['next_score_bool', 'round']][df_['round'].isin(list_idx)]



    drop_list_total = drop_list

    drop_list_total.append('round')

    set_list = set(list_idx)

    prev_round = -1

    

    for index, row in temp_df.iterrows():

        if(row['round'] > prev_round):

            prev_round = row['round']

            y_set.append(row['next_score_bool'])

    

    

    

    encoded_list = np.asarray([df_[df_['round'].isin([round_])].drop(columns=drop_list).to_numpy() for round_ in set_list])

    

    return(encoded_list, np.asarray(y_set))
train_X_test_list, train_y_test_list = encode_list_Yx9(df_total_normalized, train_idx)

test_X_test_list, test_y_test_list = encode_list_Yx9(df_total_normalized, test_idx)

validate_X_test_list, validate_y_test_list = encode_list_Yx9(df_total_normalized, validate_idx)
def null_row():

    list_ = []

    for x in range(len(df_total.drop(columns = drop_list).columns)):

        list_.append(0)

    

    return [list_]
def test_test(test_list):

    temp2 = []

    for round_ in test_list:

        if(len(round_) < 9):

            temp1 = round_

            for x in range(9 - len(round_)):    

                temp1 = np.concatenate((temp1, null_row()))

        else:

            temp1 = round_

            

        temp2.append(temp1)

    return temp2

train_X_test_list = np.asarray(test_test(train_X_test_list))

validate_X_test_list = np.asarray(test_test(validate_X_test_list))

test_X_test_list = np.asarray(test_test(test_X_test_list))
hidden_nodes = int(2/3 * (len(train_X_test_list[0][0])*9))

print(f"The number of hidden nodes is {hidden_nodes}.")
# Build the model

print('Build model...')

model = Sequential()

model.add(LSTM(hidden_nodes, return_sequences=False, input_shape=(9, len(train_X_test_list[0][0]))))

model.add(Dropout(0.2))

model.add(Dense(units=1))

model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



batch_size=100

model.fit(train_X_test_list, train_y_test_list, batch_size=32, epochs=10, validation_data=(validate_X_test_list, validate_y_test_list))



acc = model.evaluate(test_X_test_list, test_y_test_list, verbose = 2, batch_size = 32)
regressor = Sequential()

# First LSTM layer with Dropout regularisation

regressor.add(LSTM(units=hidden_nodes, return_sequences=True, input_shape=(9, len(train_X_test_list[0][0]))))

regressor.add(Dropout(0.2))

# Second LSTM layer

regressor.add(LSTM(units=hidden_nodes, return_sequences=True))

regressor.add(Dropout(0.2))

# Third LSTM layer

regressor.add(LSTM(units=hidden_nodes, return_sequences=True))

regressor.add(Dropout(0.2))

# Fourth LSTM layer

regressor.add(LSTM(units=hidden_nodes))

regressor.add(Dropout(0.2))

# The output layer            

regressor.add(Dense(units=1, activation = 'sigmoid'))



# Compiling the RNN

regressor.compile(optimizer='adam',loss='mean_squared_error', metrics=['acc'])

# Fitting to the training set

regressor.fit(train_X_test_list,train_y_test_list,epochs=50,batch_size=32, validation_data=(validate_X_test_list, validate_y_test_list))
acc = regressor.evaluate(test_X_test_list, test_y_test_list, verbose = 2, batch_size = 32)
drop_list.append('next_score_bool')

drop_list.append('next_score')

drop_list.append('round')

drop_list.append('ct_score')

drop_list.append('t_score')

train_X_test = df_total[df_total['round'].isin(train_idx)].drop(columns=drop_list)

train_y_test = df_total['next_score_bool'][df_total['round'].isin(train_idx)]



validate_X_test = df_total[df_total['round'].isin(validate_idx)].drop(columns=drop_list)

validate_y_test = df_total['next_score_bool'][df_total['round'].isin(validate_idx)]



test_X_test = df_total[df_total['round'].isin(test_idx)].drop(columns=drop_list)

test_y_test = df_total['next_score_bool'][df_total['round'].isin(test_idx)]
def encode_list_Yx1(df, list_idx):

    

    y_set = []

    

    drop_list_total = drop_list

    drop_list_total.append('round')

    set_list = set(list_idx)

    

    encoded_list = df[df['round'].isin(list_idx)].drop(columns=drop_list).to_numpy()

    

    for snapshot in encoded_list:

        snapshot = temp = [snapshot]

    

    temp_df = df[['next_score_bool', 'next_score']][df['round'].isin(list_idx)]

    for x in range(int(len(temp_df))):

        y_set.append(temp_df['next_score_bool'].iloc[x])

        

    

    return(encoded_list, np.asarray(y_set))
train_X_test_list, train_y_test_list = encode_list_Yx1(df_total_normalized, train_idx)

test_X_test_list, test_y_test_list = encode_list_Yx1(df_total_normalized, test_idx)

validate_X_test_list, validate_y_test_list = encode_list_Yx1(df_total_normalized, validate_idx)



train_X_test_list=train_X_test_list.reshape(-1, 1, len(train_X_test_list[0]))

test_X_test_list=test_X_test_list.reshape(-1, 1, len(test_X_test_list[0]))

validate_X_test_list=validate_X_test_list.reshape(-1, 1, len(validate_X_test_list[0]))
hidden_nodes = int(2/3 * (len(df_total_normalized.drop(columns=drop_list).columns)))

print(f"The number of hidden nodes is {hidden_nodes}.")
# Build the model

print('Build model...')

model = Sequential()

model.add(LSTM(hidden_nodes, return_sequences=False, input_shape=(9, len(train_X_test_list[0][0]))))

model.add(Dropout(0.2))

model.add(Dense(units=1))

model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



batch_size=100

model.fit(train_X_test_list, train_y_test_list, batch_size=batch_size, epochs=10, validation_data=(validate_X_test_list, validate_y_test_list))
acc = model.evaluate(test_X_test_list, test_y_test_list, verbose = 2, batch_size = 32)
train_X_test = df_total_normalized[df_total_normalized['round'].isin(train_idx)].drop(columns=drop_list)

train_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(train_idx)]



validate_X_test = df_total_normalized[df_total_normalized['round'].isin(validate_idx)].drop(columns=drop_list)

validate_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(validate_idx)]



test_X_test = df_total_normalized[df_total_normalized['round'].isin(test_idx)].drop(columns=drop_list)

test_y_test = df_total_normalized['next_score_bool'][df_total_normalized['round'].isin(test_idx)]
xgboost_(train_X_test, train_y_test, validate_X_test, validate_y_test)
regressor = Sequential()

# First LSTM layer with Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(1, len(train_X_test_list[0][0]))))

regressor.add(Dropout(0.2))

# Second LSTM layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

# Third LSTM layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

# Fourth LSTM layer

regressor.add(LSTM(units=50))

regressor.add(Dropout(0.2))

# The output layer

regressor.add(Dense(units=1))



# Compiling the RNN

regressor.compile(optimizer='rmsprop',loss='mean_squared_error', metrics=['acc'])

# Fitting to the training set

regressor.fit(train_X_test_list,train_y_test_list,epochs=50,batch_size=32, validation_data=(validate_X_test_list, validate_y_test_list))
acc = regressor.evaluate(test_X_test_list, test_y_test_list, verbose = 2, batch_size = 32)
xgboost_(train_X_test, train_y_test, validate_X_test, validate_y_test)
df_total['next_score_bool'].mean()
regressor = Sequential()

# First LSTM layer with Dropout regularisation

regressor.add(LSTM(units=50, return_sequences=True, input_shape=(1, len(train_X_test_list[0][0]))))

regressor.add(Dropout(0.2))

# Second LSTM layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

# Third LSTM layer

regressor.add(LSTM(units=50, return_sequences=True))

regressor.add(Dropout(0.2))

# Fourth LSTM layer

regressor.add(LSTM(units=50))

regressor.add(Dropout(0.2))

# The output layer

regressor.add(Dense(units=1))



# Compiling the RNN

regressor.compile(optimizer='adam',loss='mean_squared_error', metrics=['acc'])

# Fitting to the training set

regressor.fit(train_X_test_list,train_y_test_list,epochs=50,batch_size=32, validation_data=(validate_X_test_list, validate_y_test_list))
acc = regressor.evaluate(test_X_test_list, test_y_test_list, verbose = 2, batch_size = 32)
prev = 0;

timeouts = []

prev_freeze = False

max_i = len(df_total_normalized.index)

for index, row in df_total_normalized.iterrows():

    if(index > max_i):

        break

    elif(row['round_status'] == 'FreezeTime'):

        timeouts.append(index)



        

df_total_nofreeze = df_total_normalized.drop(df_total_normalized.index[timeouts])

df_total_nofreeze.reset_index(inplace=True, drop=True)
train_X_test_list, train_y_test_list = encode_list_Yx1(df_total_nofreeze, train_idx)

test_X_test_list, test_y_test_list = encode_list_Yx1(df_total_nofreeze, test_idx)

validate_X_test_list, validate_y_test_list = encode_list_Yx1(df_total_nofreeze, validate_idx)



train_X_test_list=train_X_test_list.reshape(-1, 1, len(train_X_test_list[0]))

test_X_test_list=test_X_test_list.reshape(-1, 1, len(test_X_test_list[0]))

validate_X_test_list=validate_X_test_list.reshape(-1, 1, len(validate_X_test_list[0]))
model = Sequential()

# First LSTM layer with Dropout regularisation

model.add(LSTM(units=50, return_sequences=True, input_shape=(1, len(train_X_test_list[0][0]))))

model.add(Dropout(0.2))

# Second LSTM layer

model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.2))

# Third LSTM layer

model.add(LSTM(units=50, return_sequences=True))

model.add(Dropout(0.2))

# Fourth LSTM layer

model.add(LSTM(units=50))

model.add(Dropout(0.2))

# The output layer

model.add(Dense(units=1))



# Compiling the RNN

model.compile(optimizer='adam',loss='mean_squared_error', metrics=['acc'])

# Fitting to the training set

model.fit(train_X_test_list,train_y_test_list,epochs=50,batch_size=32, validation_data=(validate_X_test_list, validate_y_test_list))
acc = regressor.evaluate(test_X_test_list, test_y_test_list, verbose = 2, batch_size = 32)
predictions = ['ct' if prediction[0] < 0 else 't' for prediction in model.predict(validate_X_test_list)]
predictions
prediction = model.predict(validate_X_test_list)
sum(prediction)/len(prediction)
model = Sequential()

# First LSTM layer with Dropout regularisation

model.add(LSTM(hidden_nodes, return_sequences=True, input_shape=(1, len(train_X_test_list[0][0]))))

model.add(Dropout(0.2))

# Second LSTM layer

model.add(LSTM(hidden_nodes, return_sequences=True))

model.add(Dropout(0.2))

# Third LSTM layer

model.add(LSTM(hidden_nodes, return_sequences=True))

model.add(Dropout(0.2))

# Fourth LSTM layer

model.add(LSTM(hidden_nodes))

model.add(Dropout(0.2))

# The output layer

model.add(Dense(units=1, activation='sigmoid'))



# Compiling the RNN

model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])

# Fitting to the training set

model.fit(train_X_test_list,train_y_test_list,epochs=50,batch_size=32, validation_data=(validate_X_test_list, validate_y_test_list))
pedictions = model.predict(test_X_test_list, verbose=2)
early = []

mid = []

late = []

for x in range(len(test_X_test_list)):

    if(test_X_test_list[x][0][0] > 0.8):

        if(test_y_test_list[x] == 0 and pedictions[x] < 0.5):  

            early.append(1)

        elif(test_y_test_list[x] == 1 and pedictions[x] > 0.5):

            early.append(1)

        else:

            early.append(0)

    elif(test_X_test_list[x][0][0] > -0.8):

        if(test_y_test_list[x] == 0 and pedictions[x] < 0.5):  

            mid.append(1)

        elif(test_y_test_list[x] == 1 and pedictions[x] > 0.5):

            mid.append(1)

        else:

            mid.append(0)

    else:

        if(test_y_test_list[x] == 0 and pedictions[x] < 0.5):  

            late.append(1)

        elif(test_y_test_list[x] == 1 and pedictions[x] > 0.5):

            late.append(1)

        else:

            late.append(0)

early_acc = sum(early)/len(early)

mid_acc = sum(mid)/len(mid)

late_acc = sum(late)/len(late)



print(early_acc) 

print(mid_acc) 

print(late_acc) 

        
pred = model.predict(validate_X_test_list)



pred_binary = []

for x in pred:

    if(x < 0.5):

        pred_binary.append(0)

    else:

        pred_binary.append(1)

conf_mat = confusion_matrix(validate_y_test_list, pred_binary, normalize='true')

print(conf_mat)

tn, fp, fn, tp = confusion_matrix(validate_y_test, pred_binary).ravel()
#ct precision

tp/(tp+fp)
#t precision

tn/(tn+fn)
#ct recall

tp/(tp+fn)

#t recall

tn/(tn+fp)
#harmonic f1 ct

2*0.7239675584972499*0.7133278221732341/(0.7239675584972499+0.7133278221732341)
#harmonic f1 ct

2*0.733316243698197*0.7434808975136447/(0.733316243698197+0.7434808975136447)
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
conf_mat = confusion_matrix(validate_y_test_list, pred_binary)
plot_confusion_matrix(cm           = conf_mat, 

                      normalize    = True,

                      target_names = ['ct', 't'],

                      title        = "Confusion Matrix")
stop
from hyperopt import hp, fmin, tpe, hp, STATUS_OK, Trials, space_eval

from keras.layers import Dropout, Conv1D, GlobalMaxPool1D, GRU, GlobalAvgPool1D



DROPOUT_CHOICES = np.arange(0.0, 0.9, 0.1)

UNIT_CHOICES = np.arange(8, 129, 8, dtype=int)

EMBED_UNITS = np.arange(32, 513, 32, dtype=int)

FILTER_CHOICES = list(range(1, 9, 1))

BATCH_SIZE = np.arange(10, 513, 10, dtype=int)



space = {   

    'layer_units':  hp.choice('layer_units', UNIT_CHOICES),

    'embed_units': hp.choice('embed_units', EMBED_UNITS),

    'dropout': hp.choice('dropout', DROPOUT_CHOICES),

    'batch_size':  hp.choice('batch_size', BATCH_SIZE),

    'spatial_dropout': hp.choice('spatial_dropout', DROPOUT_CHOICES),

    

    

    'conv1_units':  hp.choice('conv1_units', UNIT_CHOICES),

    'conv1_filters': hp.choice('conv1_filters', FILTER_CHOICES),

}
def objective(params, verbose=0, checkpoint_path = 'model.hdf5'):

    from keras.callbacks import ModelCheckpoint, EarlyStopping

    if verbose > 0:

        print ('Params testing: ', params)

        print ('\n ')

    

    model = Sequential()

        

    # First LSTM layer with Dropout regularisation

    model.add(LSTM(params['layer_units'], return_sequences=True, input_shape=(1, len(train_X_test_list[0][0]))))

    model.add(Dropout(params['dropout']))

    

    # Second LSTM layer

    

    model.add(SpatialDropout1D(params['spatial_dropout']))

    model.add(Conv1D(params['conv1_filters'], kernel_size = 1,input_shape=(1,train_X_test_list.shape[2])))

    

    model.add(LSTM(params['layer_units'], return_sequences=True))

    model.add(Dropout(params['dropout']))

             



    # Third LSTM layer

    model.add(LSTM(params['layer_units'], return_sequences=True))

    model.add(Dropout(params['dropout']))

        

    model.add(GlobalMaxPool1D())



     # The output layer

    model.add(Dense(units=1, activation='sigmoid'))



    

    # Compiling the RNN

    model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['acc'])



    model.fit(

        train_X_test_list, 

        train_y_test_list, 

        validation_data=(validate_X_test_list, validate_y_test_list),  

        epochs=25, 

        batch_size=params['batch_size'],

         #saves the most accurate model, usually you would save the one with the lowest loss

        callbacks= [

            ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=verbose, save_best_only=True),

            EarlyStopping(patience = 4, verbose=verbose,  monitor='val_acc')

        ],

        verbose=verbose

    ) 

    

    model.load_weights(checkpoint_path)

    acc = model.evaluate(test_X_test_list, test_y_test_list, verbose = verbose, batch_size = params['batch_size'])

    return {'loss': -acc[1], 'status': STATUS_OK} 
trials = Trials()

best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=100, rstate=np.random.RandomState(99))