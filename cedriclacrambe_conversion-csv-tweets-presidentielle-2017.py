#start by importing the necessary libraries to play with the data

import pandas as pd
import numpy as np
import sqlite3
import os
import seaborn as sns
import matplotlib.pyplot as plt
import glob
%matplotlib inline

#We need to use all sqlite files into a pandas dataframe
#We create a list to help with connection of each database
base = "../input/database_"
numListDataPath = [base+str(i)+"_"+str(j)+".sqlite" for i in range(15,19) for j in range(0,4)]

#add data for 14 and 19
numListDataPath.append('../input/database_14_3.sqlite')
numListDataPath.append('../input/database_19_0.sqlite')
numListDataPath.append('../input/database_19_1.sqlite')
numListDataPath+=glob.glob('../input/database_1*.sqlite')
numListDataPath=sorted(set(numListDataPath))

dfs_list = []

#get lang information from all tweet from all databases
for dataPath in numListDataPath:
    try:
        connection = sqlite3.connect(dataPath)
        dfs_list.append(pd.read_sql_query("SELECT * from data", connection))
        connection.close()
    except sqlite3.OperationalError as e:
        print (e)
        
    
#concat tweets of all database on one list
dfs_list = pd.concat(dfs_list)
dfs_list
dfs_list["date"]=pd.to_datetime(dfs_list.timestampms, unit='ms')
dfs_list=dfs_list.drop(["day","hour","timestampms"],axis=1)


dfs_list["lang"]=dfs_list["lang"].astype('category')
dfs_list["location"]=dfs_list["location"].astype('category')

dfs_list.head()
dfs_list.info()
import lzma
with lzma.open("tweets.txt.xz","wt",encoding="utf8") as f:
   
    for t in dfs_list["quoted/retweeted full_text"].dropna().drop_duplicates().values:
        print(t,file=f)
    for t in dfs_list.text.dropna().drop_duplicates().values:
        print(t,file=f)



dfs_list.to_csv("tweet_presidentielle1017.csv.gz",compression="gzip")

dfs_list.to_csv("tweet_presidentielle1017.csv.xz",compression='xz')
dfs_list.to_csv("tweet_presidentielle1017.csv")
