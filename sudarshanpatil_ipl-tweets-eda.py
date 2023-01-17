import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

import warnings as ws

ws.filterwarnings("ignore")
df = pd.read_csv("/kaggle/input/ipl2020-tweets/IPL2020_Tweets.csv")
df.head()
check_na = ((df.isna().sum() / df.shape[0])* 100).reset_index().rename(columns = {"index": "Columns", 0: "missing value percentage"})

fig = px.bar(check_na, y='missing value percentage', x='Columns', text='missing value percentage', title = "Percent of missing values in the columns")

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()
sns.set()

plt.figure(figsize = (10,6))

sns.barplot(data = df.user_location.value_counts()[:5].reset_index(), y = "user_location", x="index", palette="Spectral") 

plt.ylabel("Count of people")

plt.xlabel("Top Locations in the dataser")

plt.title("Top user locations in the dataset", size = 20)

plt.show()

# For the purpose of EDA we will replace the na values of location with the  india

df.user_location.fillna("India", inplace= True)
df.hashtags.fillna('[IPL2020]' , inplace = True)
df.isna().sum()
# estimate of verified user

temp = df.user_verified.replace({True: "Verified", False: "Non-verfied"}).value_counts().reset_index()

fig = px.pie(temp, values='user_verified', names='index', color_discrete_sequence=px.colors.sequential.RdBu, title = "User Status")

fig.show()



fig = px.bar(df.source.value_counts()[:10].reset_index(), y='source', x='index', text='source', title = "Top Sources of posting", color = "index")

fig.show()
fig = px.bar(df.hashtags.value_counts()[1:10].reset_index(), y='hashtags', x='index', text='hashtags', title = "Top Trending of Hashtags",  color='index')

fig.show()
fig = px.scatter(data_frame=df, y="user_favourites", x="user_followers", size = "user_favourites", color = "user_verified",log_x=True, size_max=20)

fig.show()
fig = px.bar(df.user_location.value_counts()[:20].reset_index(), y='user_location', x='index', text='user_location', title = "Top 20 locations while twitting ",  color='index')

fig.show()
df.user_created = pd.to_datetime(df.user_created, infer_datetime_format=True)

temp = pd.datetime.now() - df.user_created
avg_age = []

for i in temp:

    avg_age.append(int(str(i).split()[0]) // 365)
fig = px.bar(pd.Series(avg_age).value_counts().reset_index().rename(columns = {"index":"year", 0:"Occurances"}), x= "year", y="Occurances", color = "Occurances", title  ="Age of the twitter account ")

fig.show()
temp = df.is_retweet.value_counts().reset_index().replace({False: "Not Retweeted", True : "ReTweeted"})

fig = px.pie(temp, values='is_retweet', names='index', color_discrete_sequence=px.colors.sequential.RdBu, title = "Retweeted or not")

fig.show()