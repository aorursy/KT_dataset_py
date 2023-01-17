# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import operator, re
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
data = pd.read_csv("/kaggle/input/friends-series-dataset/friends_episodes_v2.csv")
print("Total episodes : ", data.shape[0])
data.head()
data["Episode_Title"].iloc[-1] = "The Last One Part(2)"
data.isnull().sum() # No null values
temp = data["Year_of_prod"]
print("Start :  {} \nEnd : {}".format(min(temp), max(temp)))
data[["Stars", "Votes"]].describe()
sns.distplot(data["Votes"])
plt.grid(b=None)
plt.show()
sns.distplot(data["Stars"])
plt.grid(b=None)
plt.show()
temp = data.groupby("Season").count()["Episode_Title"]
seasons = list(temp.index)
episodes = list(temp.values)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
sns.barplot(seasons,episodes, palette='inferno')
ax.grid(False)
plt.xlabel('Seasons') 
plt.ylabel('Number of episodes') 
plt.title('Episodes in a season')
plt.ylim(15, 30)
plt.show()
# Total duration of a season

duration = data.groupby("Season").sum()["Duration"]

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
sns.barplot(list(duration.index),list(duration.values), palette='inferno')
ax.grid(False)
plt.xlabel('Seasons') 
plt.ylabel('Duration') 
plt.title('Total Duration of Seasons')
plt.ylim(400, 600)
plt.show()
season_stars = data.groupby('Season').mean().Stars.to_frame()
stars = list(season_stars["Stars"])
season = list(season_stars.index)

plt.figure(figsize=(10,5))
sns.barplot(season, stars, palette='inferno')
plt.title('Average Stars of each Season')
plt.xlabel('Average Stars')
plt.ylabel('Seasons')
plt.ylim(8.2,8.8)
plt.show()
temp = data.groupby("Season").mean()["Stars"]
index = list(temp.index)
values = list(temp.values)

print("Best Season according to Stars : ", np.argmax(values)+1)
print("Least Best Season according to Stars : ", np.argmin(values)+1)
temp = data.groupby("Season").mean()["Votes"]
index = list(temp.index)
values = list(temp.values)

print("Best Season according to Votes : ", np.argmax(values)+1)
print("Least Best Season according to Votes : ", np.argmin(values)+1)
# Top 10 best episodes according to the stars

temp = data.sort_values(by=["Stars"], ascending=False)[:10]
episodes = list(temp["Episode_Title"])
stars = list(temp["Stars"])

sns.barplot(stars, episodes, palette='inferno')
plt.xlabel('Stars') 
plt.ylabel('Episodes') 
plt.title('Top 10 Episodes according to Stars')
plt.xlim(9,10)
plt.show()
# Top 10 best episodes according to the Votes

temp = data.sort_values(by=["Votes"], ascending=False)[:10]
episodes = list(temp["Episode_Title"])
votes = list(temp["Votes"])

sns.barplot(votes, episodes, palette='inferno')
plt.xlabel('Votes') 
plt.ylabel('Episodes') 
plt.title('Top 10 Episodes according to Votes')
plt.xlim(4000, 11000)
plt.show()
# Top 10 best episodes according to the Votes

temp = data.sort_values(by=["Duration"], ascending=False)[:10]
episodes = list(temp["Episode_Title"])
duration = list(temp["Duration"])

sns.barplot(duration, episodes, palette='inferno')
plt.xlabel('Duration') 
plt.ylabel('Episodes') 
plt.title('Top 10 Episodes according to Time Duration')
plt.xlim(20,40)
plt.show()
data.groupby("Director").count()["Year_of_prod"]   # Episode directed by different directory
# Top 5 High-rated Directors (Directing more than 10 Episodes)

temp = data.groupby("Director").agg({"Stars": "mean", "Season":"count"}).sort_values("Stars", ascending=False)
temp = temp[temp["Season"]>=10][:5]

stars = list(temp["Stars"])
director = list(temp.index)

sns.barplot(stars, director, palette='inferno')
plt.xlabel('Stars') 
plt.ylabel('Director') 
plt.title('Top 5 High-rated Directors (Directing more than 10 Episodes)')
plt.xlim(8,9)
plt.show()
# Top 5 High-voted work of Directors (Directing more than 10 Episodes)

temp = data.groupby("Director").agg({"Votes": "mean", "Season":"count"}).sort_values("Votes", ascending=False)
temp = temp[temp["Season"]>=10][:5]

votes = list(temp["Votes"])
director = list(temp.index)

sns.barplot(votes, director, palette='inferno')
plt.xlabel('Votes') 
plt.ylabel('Director') 
plt.title('Top 5 High-voted  work of Directors (Directing more than 10 Episodes)')
plt.xlim(3000, 4500)
plt.show()
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer 

import string
print(string.punctuation)
sid_obj = SentimentIntensityAnalyzer() 

similarity = list(map(lambda x: sid_obj.polarity_scores(x) ,data["Summary"]))
positive = {}
negative = {}

for i in range(len(similarity)):
    positive[i] = similarity[i]["pos"]
    negative[i] = similarity[i]["neg"]
positive_summary = []
temp = sorted(positive.items(), key=operator.itemgetter(1), reverse=True)
for i in range(50):
    temp1 = data["Summary"].iloc[temp[i][0]]
    temp1 = re.sub("Rachel|Ross|Chandler|Monica|Phoebe|Joey|Emily|Janice|Richard","", temp1)
    positive_summary.append(temp1)
    
negative_summary = []
temp = sorted(negative.items(), key=operator.itemgetter(1), reverse=True)
for i in range(50):
    temp1 = data["Summary"].iloc[temp[i][0]]
    temp1 = re.sub("Rachel|Ross|Chandler|Monica|Phoebe|Joey|Emily|Janice|Richard","", temp1)
    negative_summary.append(temp1)
wordcloud = WordCloud(stopwords=STOPWORDS,
                      max_words=300
                         ).generate("".join(positive_summary))

plt.figure(figsize=(10,10))
plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
wordcloud = WordCloud(stopwords=STOPWORDS, 
                      max_words=300
                         ).generate("".join(negative_summary))

plt.figure(figsize=(10,10))
plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
data["Summary"] = data["Summary"].apply(lambda x: x.lower())

summary = []
for i in data["Summary"]:
    summary.append("".join([char for char in i if char not in string.punctuation]))

data["Summary"] = summary

data["Summary"] = data["Summary"].apply(lambda x : word_tokenize(x))
data["Summary"] = data["Summary"].apply(lambda x: [word for word in x if word not in stop_words])
cast = {"Ross": 0,"Rachel": 0, "Monica": 0, "Chandler":0, "Phoebe": 0, "Joey": 0}

for i in data["Summary"]: 
    for j in i:
        if j=="ross" or j == "rosss": cast["Ross"] +=1
        elif j=="rachel" or j== "rachels" : cast["Rachel"] +=1
        elif j=="monica" or j=="monicas" : cast["Monica"] +=1
        elif j=="chandler" or j=="chandlers" : cast["Chandler"] +=1
        elif j=="phoebe" or j=="phoebes" : cast["Phoebe"] +=1
        elif j=="joey" or j=="joeys" : cast["Joey"] +=1
actors = list(cast.keys())
count = list(cast.values())

sns.barplot(count, actors, palette='inferno')
plt.xlabel('Names in Summary') 
plt.ylabel('Actors') 
plt.title('Names appears in the summary')
plt.xlim(100, 200)
plt.show()
temp = data["Stars"].sort_values(ascending=False)[:20]
index = list(temp.index)

popular_summary = []
for i in index:
    temp1 = " ".join(data["Summary"].iloc[i])
    temp1 = re.sub("rachel|ross|chandler|monica|phoebe|joey","", temp1)
    popular_summary.append(temp1)
    
wordcloud = WordCloud(stopwords=STOPWORDS,
                      max_words=300
                         ).generate("".join(popular_summary))

plt.figure(figsize=(10,10))
plt.clf()
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
