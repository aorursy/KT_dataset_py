%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

pd.set_option('display.width', 500)

pd.set_option('display.max_columns', 100)
color = sns.color_palette()

color
data=pd.read_csv("../input/top50spotify2019/top50.csv",encoding='ISO-8859-1')

data.head()
data.rename(columns={"Beats.Per.Minute":"BPM",

                     "Valence.":"Valence",

                     "Acousticness..":"Acousticness",

                     "Loudness..dB..":"Loudness",

                     "Speechiness.":"Speechiness",

                     "Track.Name":"Track",

                     "Artist.Name":"Artist"},inplace=True)
data.drop("Unnamed: 0",axis=1,inplace=True)

data.head(3)
data.info()
data=data.sort_values("Popularity", ascending=False)

data.head(3)
data.isnull().sum()

genre_counts=np.unique(data["Genre"].replace(np.NaN,'NaN',regex=True).values,return_counts=True)

genre_counts
fig, ax = plt.subplots(figsize=(5, 6))



ax.barh(range(len(np.unique(data["Genre"].values))),genre_counts[1], alpha=0.6)

ax.set_yticks(range(len(np.unique(data["Genre"].values))))

ax.set_yticklabels(np.unique(data["Genre"].values))

ax.set_title('Genre')

sns.scatterplot(data["Danceability"],data["Popularity"],color="Blue")
data.head()
data["Energy"].hist();


fig,ax = plt.subplots(figsize=(9,6))



sns.distplot(data["Energy"],label="Energy")

sns.distplot(data["Danceability"],label="Danceability")

ax.set_xlabel('')    



     

plt.legend(labels=['Energy','Danceability'],loc="upper right")

plt.show()
plt.figure(figsize=(12,8))

sns.jointplot(x=data["Danceability"], y=data['Energy'],height=10,kind="kde")

plt.ylabel('Energy', fontsize=12)

plt.xlabel("Danceability", fontsize=12)

plt.title("Danceability Vs Energy", fontsize=15)

plt.show()
artist_counts=np.unique(data["Artist"].replace(np.NaN,'NaN',regex=True).values,return_counts=True)

artist_counts
fig, ax = plt.subplots(figsize=(16, 4))



s=ax.bar(range(len(np.unique(data["Artist"].values))),artist_counts[1],alpha=0.6)

ax.set_xticks(range(len(np.unique(data["Artist"].values))))

ax.set_xticklabels(np.unique(data["Artist"].values), rotation=45, ha='right')

ax.set_title('Artist Songs Number')



for i in range(len(s)):

#     print(s[i])

    s[i].set_color(color[s[i].get_height()]);

from wordcloud import WordCloud, STOPWORDS 

string=str(data.Genre)

plt.figure(figsize=(12,8))

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='white',

                      width=1000,

                      height=1000).generate(string)

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
data.loc[data["Artist"]=="Ed Sheeran"]
data.head(1) #No. 1 song of 2019
data.loc[data["Artist"]=="Billie Eilish"]
diff=pd.concat([data.loc[data["Artist"]=="Billie Eilish"],data.loc[data["Artist"]=="Ed Sheeran"]],axis=0)

diff
fig, ax = plt.subplots(2, 2, figsize=(9, 8))

sns.barplot(diff["Artist"],diff["Danceability"],ax=ax[0][0])

sns.barplot(diff["Artist"],diff["Energy"],ax=ax[0][1])



sns.barplot(diff["Artist"],diff["Speechiness"],ax=ax[1][0])

sns.barplot(diff["Artist"],diff["BPM"],ax=ax[1][1])


