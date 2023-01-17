import pandas as pd

import requests

from geopy import geocoders

from urllib.parse import urlparse 

import tqdm

import sys

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS 

%matplotlib inline
df = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv', encoding = 'unicode_escape')

pd.set_option('display.max_columns', None)

df = df.dropna()
df.head()
labels = "acquired", "operating", "closed"

sizes = [len(df[df["status"] == "acquired"])/len(df)*100, len(df[df["status"] == "operating"])/len(df)*100, len(df[df["status"] == "closed"])/len(df)*100]

explode = (0, 0, 0.1)  # only "explode" the closed status startups



fig1, ax1 = plt.subplots(figsize=(18,8))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
labels = "USA", "Canada"

sizes = [len(df[df["country_code"] == "USA"])/len(df)*100, len(df[df["country_code"] == "CAN"])/len(df)*100]



fig1, ax1 = plt.subplots(figsize=(18,8))

ax1.pie(sizes,labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
sizes = { market : len(df[df[" market "] == market])/len(df)*100 for market in df[" market "].unique()}

labels = sorted(sizes, key=sizes.get, reverse=True)[:10]

sizes = [len(df[df[" market "] == market])/len(df)*100 for market in df[" market "].unique()]

sizes.sort(reverse=True)

sizes = sizes[:10]



fig1, ax1 = plt.subplots(figsize=(18,8))

ax1.pie(sizes,labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
data = {'acquired':  df[df["status"] == "acquired"][" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int).sum()/len( df[df["status"] == "acquired"]), 'operating': df[df["status"] == "operating"][" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int).sum()/len( df[df["status"] == "operating"]), 'closed': df[df["status"] == "closed"][" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int).sum()/len( df[df["status"] == "closed"])}

names = list(data.keys())

values = list(data.values())



fig1, ax1 = plt.subplots(figsize=(18,8))

ax1.bar(names, values)

# axs[0].bar(names, values)

# axs[2].plot(names, values)

fig1.suptitle('Categorical Plotting')
df['first_funding_at'] =  pd.to_datetime(df['first_funding_at'], format='%Y-%m-%d', errors = 'coerce')

df['last_funding_at'] =  pd.to_datetime(df['last_funding_at'], format='%Y-%m-%d')

df['funding_period'] = df['last_funding_at'] - df['first_funding_at']

df[" funding_total_usd "] = df[" funding_total_usd "].str.strip().str.replace(',', '').str.replace('-', '0').astype(int)

df = df.fillna(0)

df["funding_period"] =  df["funding_period"].astype(int)

df[" funding_total_usd "] = df[" funding_total_usd "]/len(df[" funding_total_usd "])
df.head()
df[df["status"] == "acquired"].plot.scatter(x='funding_period',y=' funding_total_usd ', figsize=(8,7), ylim=[0, 250000], xlim=[0.0, 1e18], title="Acquired startups")

df[df["status"] == "operating"].plot.scatter(x='funding_period',y=' funding_total_usd ', figsize=(8,7), ylim=[0, 250000], xlim=[0.0, 1e18], title="Operating startups")

df[df["status"] == "closed"].plot.scatter(x='funding_period',y=' funding_total_usd ', figsize=(8,7), ylim=[0, 250000], xlim=[0.0, 1e18], title="Closed startups")
grant_df = df[df['grant'] != 0]



labels = "acquired", "operating", "closed"

sizes = [len(grant_df[grant_df["status"] == "acquired"])/len(grant_df)*100, len(grant_df[grant_df["status"] == "operating"])/len(grant_df)*100, len(grant_df[grant_df["status"] == "closed"])/len(grant_df)*100]

explode = (0, 0, 0.1)  # only "explode" the closed status startups



fig1, ax1 = plt.subplots(figsize=(18,8))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
grant_cat = grant_df[' market ']

grant_cats = grant_cat.tolist()

unique_string=(" ").join(grant_cats)

wordcloud = WordCloud(width = 1000, height = 500).generate(unique_string)

plt.figure(figsize=(15,8))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
city_wise_spread = pd.DataFrame.from_dict({"city":[],"count":[], "acquired-count":[],"operating-count":[], "closed-count":[]})



for city, group in df.groupby("city"):

    data = [city, len(group), len(group[group["status"] == "acquired"]), len(group[group["status"] == "operating"]), len(group[group["status"] == "closed"])]

    city_wise_spread = city_wise_spread.append(pd.Series(data, index=["city","count", "acquired-count","operating-count", "closed-count"] ), ignore_index=True)



city_wise_spread.sort_values(by=['count'], ascending=False, inplace=True)
city_wise_spread.head()
sample = city_wise_spread[:10]

labels = sample["city"]

sizes = [value for value in sample["count"]]





fig1, ax1 = plt.subplots(figsize=(18,8))

ax1.pie(sizes,labels=labels, autopct='%1.1f%%', startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()