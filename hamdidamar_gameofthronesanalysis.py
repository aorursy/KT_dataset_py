import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import seaborn as sns

import sys

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS
battles = pd.read_csv("/kaggle/input/game-of-thrones/battles.csv")

deaths = pd.read_csv("/kaggle/input/game-of-thrones/character-deaths.csv")

predictions = pd.read_csv("/kaggle/input/game-of-thrones/character-predictions.csv")
battles.head()
deaths.head()
battles.columns
deaths.columns
battles.info()
chosen_columns_battles =['name', 'year', 'battle_number', 'attacker_king', 'defender_king','attacker_outcome',

       'battle_type', 'major_death', 'major_capture', 'attacker_size',

       'defender_size', 'attacker_commander', 'defender_commander', 'summer',

       'location', 'region']

df_battles = pd.DataFrame(battles, columns = chosen_columns_battles)

df_battles.set_index('name', inplace=True)

df_battles.head()
df_battles['attacker_king'].value_counts().head()
df_battles['defender_king'].value_counts().head()
df_battles['battle_type'].value_counts().head()
df_battles['attacker_commander'].value_counts().head()
df_battles['defender_commander'].value_counts().head()
df_battles['summer'].value_counts().head() 
df_battles['location'].value_counts().head()
df_battles['region'].value_counts().head()
df_battles['attacker_size'].value_counts().head()
df_battles['defender_size'].value_counts().head()
deaths.info()
chosen_columns_deaths =['Name', 'Allegiances', 'Death Year', 'Book of Death', 'Death Chapter',

       'Book Intro Chapter', 'Gender', 'Nobility', 'GoT', 'CoK', 'SoS', 'FfC',

       'DwD']

df_deaths = pd.DataFrame(deaths, columns = chosen_columns_deaths)

df_deaths.set_index('Name', inplace=True)

df_deaths.head()
battles.describe()
deaths.describe()
sns.countplot(deaths['Allegiances'])

plt.xticks(Rotation = 90)

plt.title('Allegiances Göre Ölüm ')

plt.show()
sns.countplot(deaths['Gender'])

plt.title('Cinsiyete Göre Ölüm')

plt.xticks(np.arange(2),('Kadın','Erkek'))

plt.show()
plt.figure(figsize=(10,10))

plt.title('Saldırı Krallar')

plt.subplot(2,1,1)

sns.countplot(x='attacker_king',data = battles)

plt.show()



plt.figure(figsize=(15,10))

plt.subplot(2,1,2)

sns.countplot(x='defender_king',data = battles)

plt.show()
sns.countplot(hue= battles['battle_type'],x=battles['attacker_king'],palette = 'Set3')

plt.title('Saldırgan Krallar Tarafından Kullanılan Savaş Türü ')

plt.legend(loc = 'upper right')

plt.xticks(rotation = 90)

plt.show()
sns.countplot(x='summer',data = battles)

plt.title('Mevsimlere Göre Savaşların Sayısı')

plt.xticks(np.arange(2),('Kış','Yaz'))

plt.show()
sns.countplot(x='region', data = battles)

plt.title('Bölgelere Göre Savaşların Sayısı')

plt.xticks(rotation = 90)

plt.show()
sns.countplot(x='region',hue='attacker_king', data = battles)

plt.title('Hangi kral hangi bölgeye saldırdı')

plt.xticks(rotation = 90)

plt.legend(loc = 'upper right')

plt.show()
deaths.hist()

plt.show()
co = deaths.corr()

sns.heatmap(co, annot=True, linewidths=1.0)
most_pop_genres = battles['defender_commander'].str.cat(sep=', ').split(', ')

most_pop_genres = pd.Series(most_pop_genres).value_counts() 

graph = most_pop_genres.plot.bar()

graph.set_title("En Savunmacı Kumandan", fontsize=18, fontweight='bold')

graph.set_xlabel("Savaş Sayıları", fontsize=16)

graph.set_ylabel("Kumandan Listesi", fontsize=16)

graph.set_xlim(right=10)

graph.legend(['Kumandan'], loc = "upper right")
most_pop_genres = battles['attacker_commander'].str.cat(sep=', ').split(', ')

most_pop_genres = pd.Series(most_pop_genres).value_counts(ascending=False) 

graph = most_pop_genres.plot.bar()

graph.set_title("En Saldırgan Kumandan", fontsize=18, fontweight='bold')

graph.set_xlabel("Savaş Sayıları", fontsize=16)

graph.set_ylabel("Kumandan Listesi", fontsize=16)

graph.set_xlim(right=10)

graph.legend(['Kumandan'], loc = "upper right")
#region,location,defender_commander,attacker_commander

plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='black',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_battles['region']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()



plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='Red',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_battles['location']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()



plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='Purple',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_battles['defender_commander']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()



plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='Yellow',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_battles['attacker_commander']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()



plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='Green',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_battles['attacker_king']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()



plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='White',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_battles['defender_king']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()



plt.figure(figsize = (30,30))

stopwords = set(STOPWORDS)

wordcloud = WordCloud(

                          background_color='Grey',

                          stopwords=stopwords,

                          max_words=1200,

                          max_font_size=120, 

                          random_state=42

                         ).generate(str(df_battles['battle_type']))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.title("WORD CLOUD - TAGS")

plt.axis('off')

plt.show()