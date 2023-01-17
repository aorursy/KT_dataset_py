import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns





import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")

df.head()
df.tail()
df.isnull().values.any()
df.info()
df.date=pd.to_datetime(df.date)
df.dtypes
df.head()
df.shape
df.describe().T
df.nunique()
df["winner_home"]=df.home_score > df.away_score  

df["winner_away"]=df.away_score > df.home_score

df["loser_home"]=df.home_score < df.away_score

df["loser_away"]=df.away_score < df.home_score

df["scoreless"]=df.home_score == df.away_score

df=df.replace(True,1)

df=df.replace(False,0)



df.head()
df.neutral = df.neutral.astype(int)

df.winner_home = df.winner_home.astype(int)

df.winner_away = df.winner_away.astype(int)

df.loser_home = df.loser_home.astype(int)

df.loser_away = df.loser_away.astype(int)

df.scoreless = df.scoreless.astype(int)

df.head()
df.winner_home[df.winner_home==1]=df.home_team

df.winner_home[df.home_score==df.away_score]="Scoreless"

df.winner_home[df.winner_home==0]=df.away_team



df.loser_home[df.loser_home==1]=df.home_team

df.loser_home[df.home_score==df.away_score]="Scoreless"

df.loser_home[df.loser_home==0]=df.away_team

df.head(5)

df.drop(['winner_away', 'loser_away'], axis=1,inplace=True)

df.head()
df.rename(columns={'winner_home':'winner',"loser_home":"loser"}, inplace=True)

df.head()
df = df.sort_values(by=["winner","home_score"], ascending=False)

df['rank']=tuple(zip(df.winner,df.home_score))

df['rank']=df.groupby('winner',sort=False)['rank'].apply(lambda x : pd.Series(pd.factorize(x)[0])).values

df.head()
df.reset_index(inplace=True,drop=True)

df.head()
plt.figure(figsize=(18,6))

ax = df.home_score[:750].plot.kde()

ax = df.away_score[:750].plot.kde()

ax.legend()

plt.show()
fig=plt.figure(figsize=(20,10))

sns.boxplot(df.winner[:750],df.home_score[:750])

plt.xticks(rotation=90)

plt.show()
fig=plt.figure(figsize=(20,10))

sns.boxplot(df.winner[:750],df.away_score[:750])

plt.xticks(rotation=90)

plt.show()
fig=plt.figure(figsize=(20,10))

sns.boxplot(df.loser[:750],df.home_score[:750])

plt.xticks(rotation=90)

plt.show()
fig=plt.figure(figsize=(20,20))

sns.boxplot(df.loser[:750],df.away_score[:750])

plt.xticks(rotation=90)

plt.show()
fig=plt.figure(figsize=(10,5))

sns.set(style="whitegrid", palette="muted")

sns.swarmplot(x=df.home_score[:750], y=df.away_score[:750], hue=df.scoreless[:750],

              palette=["r", "c", "y"], data=df)

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text= " ".join(str(each) for each in df.tournament)



wordcloud=WordCloud(max_words=50,background_color="white").generate(text)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text= " ".join(str(each) for each in df.country)



wordcloud=WordCloud(max_words=50,background_color="white").generate(text)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text= " ".join(str(each) for each in df.city)



wordcloud=WordCloud(max_words=50,background_color="white").generate(text)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text= " ".join(str(each) for each in df.loser)



wordcloud=WordCloud(max_words=50,background_color="white").generate(text)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
from wordcloud import WordCloud, ImageColorGenerator

text= " ".join(str(each) for each in df.winner)



wordcloud=WordCloud(max_words=50,background_color="white").generate(text)

plt.figure(figsize=(20,10))

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis("off")

plt.show()
#i tried to hundred times from five thousand to eight hundred and seventy-five and best visualization is 875

labels = df.winner[:875].value_counts().index

colors = ["black","gray","silver","whitesmoke","rosybrown",

          "firebrick","red","darksalmon","sienna"]

explode = [0,0,0,0,0,

           0,0,0,0]

sizes = df.winner[:875].value_counts().values



# visual 

plt.figure(0,figsize = (18,18))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Teams According to Win',color = 'blue',fontsize = 15)

plt.show()