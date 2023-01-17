import pandas as pd
import io
import requests
import matplotlib.pyplot as plt 
import seaborn as sns

df =pd.read_csv("../input/iron-march-fascist-social-network-dataset/csv/core_members.csv")
df
df.member_id.size
df_filter=df.filter(['member_id', 'name', 'email', 'joined', 'member_posts', 'bday_year','timezone'], axis =1)
df_filter
df_filter.sort_values(by=['member_posts'], ascending = False)
df_filter['timezone'].value_counts(dropna=True)
time = df_filter['timezone'].value_counts(dropna=True).head(10)
labels=time.index
plt.figure(figsize=(15,8))
sns.barplot(x=time, y=labels)
df_filter.groupby(['timezone'])['member_posts'].sum().reset_index().sort_values(by=['member_posts'], ascending = False)
active = df_filter.groupby(['timezone'])['member_posts'].sum().reset_index().sort_values(by=['member_posts'], ascending = False).head(10)
labels =active['timezone']
data = active['member_posts']
plt.figure(figsize=(15,8))
sns.barplot(x=data, y=labels)
df_modlog =pd.read_csv("../input/iron-march-fascist-social-network-dataset/csv/core_moderator_logs.csv")
df_modlog
df_mods = df_modlog.filter(['member_id', 'member_name']).dropna().drop_duplicates(subset=['member_id'], keep='last').sort_values(by=['member_id'], ascending = True)
df_mods
df1 = df_filter.filter(['member_id', 'member_posts', 'timezone'])
df1
#top10mods
df_mods = df_mods.merge(df1).sort_values(by=['member_posts'], ascending = False).head(10)
df_mods
