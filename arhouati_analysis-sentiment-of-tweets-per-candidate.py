#start by importing the necessary libraries to play with the data

import pandas as pd
import numpy as np
import sqlite3

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#We need to use all sqlite files into a pandas dataframe
#We create a list to help with connection of each database
base = "../input/database_"
numListDataPath = [base+str(i)+"_"+str(j)+".sqlite" for i in range(15,19) for j in range(0,4)]

#add data for 14 and 19
numListDataPath.append('../input/database_14_3.sqlite')
numListDataPath.append('../input/database_19_0.sqlite')
numListDataPath.append('../input/database_19_1.sqlite') 

dfs_list = []

#get lang information from all tweet from all databases
for dataPath in numListDataPath:
    connection = sqlite3.connect(dataPath)
    dfs_list.append(pd.read_sql_query("SELECT lang from data", connection))
    connection.close()
    
#concat tweets of all database on one list
dfs_list = pd.concat(dfs_list)

#Make a graph for to see the repartition
fig, ax = plt.subplots(figsize=(12,12))
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_xlabel('Language', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 languages of all tweets', fontsize=15, fontweight='bold')

# count of number of tweets per lang
lang_count = dfs_list["lang"].value_counts().nlargest(5)

#plot the graph
lang_count.plot(ax=ax, kind='bar', color='red')
candidate_count_tmp = []

#get lang information from all tweet from all databases
for dataPath in numListDataPath:
    connection = sqlite3.connect(dataPath)
    candidate_count_tmp.append(pd.read_sql_query("SELECT sum(mention_Arthaud) as Arthaud, sum(mention_Asselineau) as Asselineau,sum(mention_Cheminade) as Cheminade,sum(`mention_Dupont-Aignan`) as `Dupont-Aignan`,sum(mention_Fillon) as Fillon, sum(mention_Hamon) as Hamon,sum(mention_Lassalle) as Lassalle, sum(`mention_Le Pen`) as `Le Pen`,sum(mention_Macron) as Macron,sum(mention_Mélenchon) as Mélenchon,sum(mention_Poutou) as Poutou from data where lang = 'fr'", connection))
    connection.close()
    
# count number of tweets per candidate for all period
candidate_count = sum(candidate_count_tmp)

#Make a graph for to see the repartition
fig, ax = plt.subplots(figsize=(12,12))
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_xlabel('Candidates', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('Top 5 mentioned candidates of all tweets', fontsize=15, fontweight='bold')

#plot the graph
candidate_count.plot(ax=ax, kind='bar')
candidate_tweets_day = []

#get number of tweets per candidate and day
for dataPath in numListDataPath:
    connection = sqlite3.connect(dataPath)
    candidate_tweets_day.append(pd.read_sql_query("SELECT day,sum(mention_Fillon) as Fillon,sum(mention_Hamon) as Hamon,sum(`mention_Le Pen`) as `Le Pen`,sum(mention_Macron) as Macron, sum(mention_Mélenchon) as Mélenchon from data where lang = 'fr' group by day ORDER BY day  ASC", connection))
    connection.close()

# preprocessing result
candidate_tweets_day = pd.concat(candidate_tweets_day)
candidate_tweets_day = candidate_tweets_day.groupby(['day']).sum()

# Table of number tweets per day and candidate 
candidate_tweets_day

# The evolution of tweets per candidate and days
fig, ax = plt.subplots(figsize=(10,5))
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_xlabel('Days', fontsize=15)
ax.set_ylabel('Number of tweets' , fontsize=15)
ax.set_title('The evolution of tweets number per candidate', fontsize=15, fontweight='bold')

candidate_tweets_day.plot(ax=ax)
distinct_user_per_day = []

#get number of dinstinct user per day
for dataPath in numListDataPath:
    connection = sqlite3.connect(dataPath)
    distinct_user_per_day.append(pd.read_sql_query("SELECT distinct(day), count(distinct(user)) as 'distinct user' from data where lang = 'fr' group by day ORDER BY day  ASC", connection))
    connection.close()

# preprocessing result
distinct_user_per_day = pd.concat(distinct_user_per_day)
distinct_user_per_day = distinct_user_per_day.groupby(['day']).sum()

distinct_user_per_day

# The evolution of distinct user per days
fig, ax = plt.subplots(figsize=(10,5))
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_xlabel('Days', fontsize=15)
ax.set_ylabel('Number of distinct users' , fontsize=15)
ax.set_title('The evolution of all distinct user', fontsize=15, fontweight='bold')

distinct_user_per_day.plot(ax=ax)

distinct_user_per_day_candidat = []

#get number of dinstinct user per day
for dataPath in numListDataPath:
    connection = sqlite3.connect(dataPath)
    distinct_user_per_day_candidat.append(pd.read_sql_query("SELECT day, count(distinct(user)) as fillon from data where lang = 'fr' and mention_Fillon = 1 group by day ORDER BY day ASC", connection))
    distinct_user_per_day_candidat.append(pd.read_sql_query("SELECT day, count(distinct(user)) as Hamon from data where lang = 'fr' and mention_Hamon = 1 group by day ORDER BY day ASC", connection))
    distinct_user_per_day_candidat.append(pd.read_sql_query("SELECT day, count(distinct(user)) as `Le Pen` from data where lang = 'fr' and `mention_Le Pen` = 1 group by day ORDER BY day ASC", connection))
    distinct_user_per_day_candidat.append(pd.read_sql_query("SELECT day, count(distinct(user)) as Macron from data where lang = 'fr' and mention_Macron = 1 group by day ORDER BY day ASC", connection))
    distinct_user_per_day_candidat.append(pd.read_sql_query("SELECT day, count(distinct(user)) as Mélenchon from data where lang = 'fr' and mention_Mélenchon = 1 group by day ORDER BY day ASC", connection))
    connection.close()

# preprocessing result
distinct_user_per_day_candidat = pd.concat(distinct_user_per_day_candidat)
distinct_user_per_day_candidat = distinct_user_per_day_candidat.groupby(['day']).sum()

# Table of number tweets per day and candidate 
distinct_user_per_day_candidat

# The evolution of distinct user per days
fig, ax = plt.subplots(figsize=(10,5))
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
ax.set_xlabel('Days', fontsize=15)
ax.set_ylabel('Number of distinct users' , fontsize=15)
ax.set_title('The evolution of distinct user per candidate', fontsize=15, fontweight='bold')

distinct_user_per_day_candidat.plot(ax=ax)
#candidates_tags={
 #   "Hamon":["@benoithamon","@AvecHamon2017","@partisocialiste"],
#    "Le Pen":["@MLP_officiel","@FN_officiel"],
#    "Macron":["@EmmanuelMacron","@enmarchefr"],
#    "Mélenchon":["@JLMelenchon","@jlm_2017"],
#    "Fillon":["@FrancoisFillon","@Fillon2017_fr","@lesRepublicains"]
#}
#Setup the keyword dictionnary for the filter part (used in the python script to filter the twitter streaming flow)
#candidates_keywords={}
#for candidate in candidates_tags:
#    list_twitter_accounts=candidates_tags[candidate]
#    candidates_keywords[candidate]=list_twitter_accounts+[candidate]

#pprint(candidates_keywords)

_#Data collection
#block kaggle kernel
#connection = sqlite3.connect("../input/database.sqlite")

# block local kernel
#list_files=["database_11_1.sqlite","database_12_0.sqlite","database_12_1.sqlite","database_13_0.sqlite","database_13_1.sqlite","database_14_0.sqlite"]

#list_df=[]
#for file in list_files:
#    print(file)
#    connection = sqlite3.connect("../input/{}".format(file))
#    df_tweet= pd.read_sql_query("SELECT * from data", connection)
#    connection.close()
    #Filter the data 
#    df_tweet=df_tweet.loc[df_tweet['lang']=="fr"]
#    list_df.append(df_tweet)

#df_tweets=pd.concat(list_df,axis=0)
#print(df_tweets.head())

