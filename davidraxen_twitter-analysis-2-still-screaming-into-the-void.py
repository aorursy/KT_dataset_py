#Importing some packages.



import json



import string

import re



#Data visualization

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sns



#Date

import datetime as dt

import time



from math import pi

import os



import pandas as pd

import numpy as np
#Read tweets from .js-file

f = open("../input/tweet.js", "r")

f = f.read()[24:]

twitter_data = json.loads(f)



#Create a dataframe from the dictionaries



df = pd.DataFrame()

for dictionary in twitter_data:

    df = df.append(pd.DataFrame.from_dict([dictionary[list(dictionary.keys())[0]]]), sort=False)

del dictionary, f
#keys to delete

keys = ['retweeted','display_text_range', 'id_str','truncated','id', 'possibly_sensitive','favorited','extended_entities', 'in_reply_to_status_id', 'in_reply_to_status_id_str','in_reply_to_user_id', 'in_reply_to_user_id_str']

df = df.drop(keys, axis=1)

del keys

   

#- Create som plotable columns from the date.    

df["created_at"] = pd.to_datetime(df['created_at'])



df["Weekday"] = df["created_at"].dt.weekday

df["Year"] = df["created_at"].dt.year

df["Month"] = df["created_at"].dt.month

df["Day"] = df["created_at"].dt.day

df["Hour"] = df["created_at"].dt.hour



#Cleaning up the source-column

pattern = "\>(.*?)\<"

df["source"] = df["source"].apply(lambda x: re.search(pattern, x).group(1))

#- Remove mentions and links from tweets.

df["full_text"] = df["full_text"].apply(lambda x:

                                        " ".join(filter(lambda y:y[0]!="@" and y[0:4]!="http", x.split())))





#Creating a column with the number of characters for each tweet without the mentions.



df["no_characters"] = df["full_text"].apply(lambda x: len(x))





#----- Make integers out of strings in numerical columns

df['favorite_count'] = df['favorite_count'].astype(int)

df['retweet_count'] = df['retweet_count'].astype(int)



#------ Create a for RTs and not RTs

df["RTs"] = df["full_text"].apply(lambda x: 1 if 

                                  " ".join(filter(lambda y:y[:2]=="RT", x.split())) else 0)

plt.figure(figsize=(10,10))

df.Weekday.value_counts().plot.pie(autopct='%1.1f%%',shadow=True,cmap='brg')

plt.figure(figsize=(15,10))

g = sns.countplot(x="Hour", data=df, order=list(range(0,24)))

g.set_xticklabels(g.get_xticklabels(),rotation=30)
plt.figure(figsize=(15,10))

g = sns.countplot(x="Day", data=df, order=list(range(1,32)))

g.set_xticklabels(g.get_xticklabels(),rotation=30)
Years = df["Year"].unique().tolist()

df_temp = df.groupby(["Month"]).apply(lambda column: column["Year"].value_counts()).unstack().reset_index()

df_temp.plot(x="Month", y=Years, kind="bar", figsize=(20, 10))
df_temp = df.groupby(["source"]).apply(lambda column: column["Year"].value_counts()).unstack().reset_index()

df_temp.plot(x="source", y=Years, kind="bar", figsize=(20, 10))

plt.xticks(rotation=30)
plt.figure(figsize=(10,10))

sns.countplot(x="Year", data=df)
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



sns.countplot(x="favorite_count", data=df, order=df['favorite_count'].value_counts().index, ax=ax1)

plt.xlabel("Favorites")

sns.countplot(x="lang", data=df, order=df['lang'].value_counts().iloc[:5].index, ax=ax2)

plt.xlabel("Language")

sns.countplot(x="retweet_count", data=df, order=df['retweet_count'].value_counts().index, ax=ax3)

plt.xlabel("Retweets")

sns.countplot(x="in_reply_to_screen_name", data=df, order=df["in_reply_to_screen_name"].value_counts().iloc[:10].index, ax=ax4)

plt.xticks(rotation=30)

del ax1, ax2, ax3, ax4, fig
#Just to show that information can be extracted from the entities-column as well

#I'm going to pull out the chosen username connected to the screen name



df["user_mentions"] = df["entities"].map(lambda x: x["user_mentions"])

def f(user_mentions, in_reply_to_screen_name):

    if len(user_mentions) != 0:

        for d in user_mentions:

            if d["screen_name"] == in_reply_to_screen_name:

                return d["name"]

    return np.nan

        

df["user"] = df.apply(lambda x: f(x.user_mentions, x.in_reply_to_screen_name), axis=1)



plt.figure(figsize=(10,10))

sns.countplot(x="user", data=df, order=df["user"].value_counts().iloc[:10].index)

plt.xticks(rotation=30)
plt.figure(figsize=(10,10))

df.RTs.value_counts().plot.pie(autopct='%1.1f%%',shadow=True,cmap='brg')
sns.pairplot(df, x_vars=['no_characters','Weekday',

       'Year', 'Month', 'Day', 'Hour']

                    , y_vars=['favorite_count', 'retweet_count']

                    , kind="reg" ,diag_kind = 'hist' );