# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#System libraries

import datetime

from pprint import pprint

from itertools import product





#Data analytics libraries

import numpy as np

import pandas as pd



#Data management libraries

import sqlite3



#Plot libraries

%matplotlib inline

from matplotlib import patheffects

import matplotlib.pyplot as plt
#Candidate informations space

candidates_tags={

    "Hamon":["@benoithamon","@AvecHamon2017","@partisocialiste"],

    "Le Pen":["@MLP_officiel","@FN_officiel"],

    "Macron":["@EmmanuelMacron","@enmarchefr"],

    "Mélenchon":["@JLMelenchon","@jlm_2017"],

    "Fillon":["@FrancoisFillon","@Fillon2017_fr","@lesRepublicains"],

    "Dupont-Aignan":["@dupontaignan","@DLF_Officiel"],

    "Cheminade":["@JCheminade"],

    "Arthaud":["@n_arthaud","@LutteOuvriere"],

    "Asselineau":["@UPR_Asselineau","@UPR_Officiel"],

    "Poutou":["@PhilippePoutou","@NPA_officiel"],

    "Lassalle":["@jeanlassalle"]

}

#Setup the keyword dictionnary for the filter part (used in the python script to filter the twitter streaming flow)

candidates_keywords={}

for candidate in candidates_tags:

    list_twitter_accounts=candidates_tags[candidate]

    candidates_keywords[candidate]=list_twitter_accounts+[candidate]



pprint(candidates_keywords)
# block local kernel

list_files=["database_11_1.sqlite","database_12_0.sqlite","database_12_1.sqlite","database_13_0.sqlite","database_13_1.sqlite","database_14_0.sqlite","database_14_1.sqlite","database_15_0.sqlite","database_15_1.sqlite","database_16_0.sqlite","database_16_1.sqlite"]

list_df=[]

for file in list_files:

    print(file)

#     connection = sqlite3.connect("../../../../Data/twitter_pres2017/{}".format(file))

    connection = sqlite3.connect("../input/{}".format(file))

    df_tweet= pd.read_sql_query("SELECT * from data", connection)

    connection.close()

    #Filter the data 

    df_tweet=df_tweet.loc[df_tweet['lang']=="fr"]

    df_tweet["count_tweets"]=[1]*len(df_tweet)

    

    

    list_columns=["day","count_tweets"]

    for candidate in candidates_tags:

        list_columns.append("mention_"+candidate)

    df_mention=df_tweet[list_columns]

#     print(df_mention.head())

    df_mentions_perday=df_mention.groupby(["day"]).sum()

    

    del list_columns[:2]

    for col in list_columns:

        df_mentions_perday[col]=df_mentions_perday.apply(lambda row : 100*row[col]/row["count_tweets"],axis=1)

        

    print(df_mentions_perday.head())

    list_df.append(df_mentions_perday)

    

    

    

df_tot=pd.concat(list_df,axis=0)









print("FINAL_DF:",df_tot.head())

    

    

    

    
def plot_heatmap(df,title):

    fig, ax = plt.subplots(figsize=(20,20))

    m, n = len(df.index),len(df.columns)

    ax.tick_params(axis='x', labelsize=10)

    ax.tick_params(axis='y', labelsize=10)

    ax.set_xlabel(title, fontsize=15)

    #ax.set_ylabel('Candidates' , fontsize=15)

    #ax.set_title('Who with who', fontsize=15, fontweight='bold')

    ax = plt.imshow(df, interpolation='nearest', cmap='seismic').axes



    _ = ax.set_xticks(np.linspace(0, n-1, n))

    _ = ax.set_xticklabels(df.columns,rotation=45)

    _ = ax.set_yticks(np.linspace(0, m-1, m))

    _ = ax.set_yticklabels(df.index)





    ax.grid('off')

    ax.xaxis.tick_top()

    path_effects = [patheffects.withSimplePatchShadow(shadow_rgbFace=(1,1,1))]



    for i, j in product(range(m), range(n)):

        _ = ax.text(j, i, '{0:.2f}'.format(df.iloc[i, j]),

            size='medium', ha='center', va='center',path_effects=path_effects)

    

    return fig,ax
list_columns=[]

for candidate in candidates_tags:

    list_columns.append("mention_"+candidate)



df_toplot=df_tot[list_columns].copy()  

    

clean_names=[ col.split("_")[1] for col in list_columns]

df_toplot.columns=clean_names

    

fig,ax=plot_heatmap(df_toplot,"Percentaga mentions in the tweets during the day")

# ax.annotate('Big debate and Poutou talk', xy=(12,16), xytext=(12,16),

#             arrowprops=dict(facecolor='black', shrink=0.05),

#             )





# connection = sqlite3.connect("../../../../Data/twitter_pres2017/database_googletrends.sqlite")

file="database_googletrends.sqlite"

connection = sqlite3.connect("../input/{}".format(file))

df_interests= pd.read_sql_query("SELECT * from interests_over_time", connection)

connection.close()

#Filter the data

del df_interests["index"]

df_interests.index=df_interests["day"]

del df_interests["day"]

print(df_interests.head())
correspondance_columns={

    "MACRON":["emmanuel macron","Macron"],

    "FILLON":["francois fillon","Fillon"],

    "LE PEN":["marine le pen","Le Pen"],

    "MELENCHON":["jean-luc melenchon","Mélenchon"],

    "HAMON":["benoit hamon","Hamon"]

}
dict_analytic={}

for candidate in correspondance_columns:

    print(candidate)

    data_twitter=df_toplot[correspondance_columns[candidate][1]]

    

    data_google=df_interests[correspondance_columns[candidate][0]]

#     print(data_twitter.head(),data_google.head())

    

    df_data=pd.concat([data_twitter,data_google],axis=1)

    df_data.columns=["twitter_%mention","Google_interests"]

    df_data=df_data.dropna()

    

    df_data["distance"]=df_data.apply(lambda row: np.sqrt((np.power(row["twitter_%mention"],2)+ np.power(row["Google_interests"],2))),axis=1)

    

    

#     print(df_data.head())

    

#     fig, ax = plt.subplots(figsize=(12,12))

#     ax.tick_params(axis='x', labelsize=10)

#     ax.tick_params(axis='y', labelsize=10)

    

#     ax.set_xlim([0,100])

#     ax.set_ylim([0,100])

    

#     ax.set_xlabel('twitter_%mention', fontsize=15)

#     ax.set_ylabel('Google_interests' , fontsize=15)

#     ax.set_title('Mention VS interests for {}'.format(candidate), fontsize=15, fontweight='bold')



#     df_data.plot(ax=ax,kind='scatter', x='twitter_%mention', y='Google_interests');



    

    dict_analytic[candidate]=df_data
fig, ax = plt.subplots(figsize=(12,12))

ax.tick_params(axis='x', labelsize=10)

ax.tick_params(axis='y', labelsize=10)



ax.set_xlim([0,100])

ax.set_ylim([0,100])



ax.set_xlabel('twitter_%mention', fontsize=15)

ax.set_ylabel('Google_interests' , fontsize=15)

ax.set_title('Mention VS interests', fontsize=15, fontweight='bold')





color_id=["Red","Blue","Yellow","Pink","Black"]

for i,candidate in enumerate(dict_analytic):

    dict_analytic[candidate].plot(ax=ax,kind='scatter', x='twitter_%mention', y='Google_interests', label=candidate,color=color_id[i]);
fig, ax = plt.subplots(figsize=(12,12))

ax.tick_params(axis='x', labelsize=10)

ax.tick_params(axis='y', labelsize=10)







ax.set_xlabel('day', fontsize=15)

ax.set_ylabel('distance_interests' , fontsize=15)

ax.set_title('Interests', fontsize=15, fontweight='bold')





color_id=["Red","Blue","Yellow","Pink","Black"]

list_df=[]

list_cand=[]

for i,candidate in enumerate(dict_analytic):

    dict_analytic[candidate].plot(ax=ax, y='distance', label=candidate,color=color_id[i])

    list_df.append(dict_analytic[candidate]['distance'])

    list_cand.append(candidate)

    

    

df_glob=pd.concat(list_df,axis=1)

df_glob.columns=list_cand

print(df_glob.head())



df_glob["tot_distance"]=df_glob.sum(axis = 1)

print(df_glob.head())
for candidate in list_cand:

    df_glob[candidate]=df_glob.apply(lambda row : 100*row[candidate]/row["tot_distance"],axis=1)

print(df_glob.head())

del df_glob["tot_distance"]

    

    
fig, ax = plt.subplots(figsize=(12,12))

ax.tick_params(axis='x', labelsize=10)

ax.tick_params(axis='y', labelsize=10)

df_glob.plot(ax=ax,kind='bar',stacked=True)

plt.show()
print(df_glob.iloc[-1])