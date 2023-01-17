#install japanese fonts.

!pip install japanize-matplotlib
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#japanese setting.

import seaborn as sns

sns.set(font="IPAexGothic")



#Load a library for graphing.

import matplotlib.pyplot as plt

import japanize_matplotlib



import calendar

from datetime import date



from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



from janome.tokenizer import Tokenizer

from janome.analyzer import Analyzer

from janome.charfilter import *

from janome.tokenfilter import *
#Read csv.

df = pd.read_csv('/kaggle/input/kojima-tweets/kojima_tweets.csv')
#Remove date and fraction from time data.

df['date'] = pd.to_datetime(df['Created Date'])

df['hour'] = df['date'].dt.hour
#Add Tweet Count.

df['Tweets'] = 1
#Plot Tweets Trend.

df2 = df[['date', 'Tweets']]

df2.set_index('date', inplace=True)

df2 = df2.resample('W').sum() 



df2['Tweets'].plot(label="Tweet count")

plt.title('Tweets Trend')

plt.legend(ncol=1)

plt.show()
graph = df[['No of Hashtags', 'No of User Mentions', 'No of URLS added', 'No of Media added']].sum()

graph.plot.bar(figsize=(8,7))

plt.title('Tweets Content')



for x, y in zip(np.arange(4), graph.values):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
graph = df[['Media Type', 'Tweets']].groupby('Media Type').sum()

graph = graph['Tweets'].astype(int)

graph.plot.bar(figsize = (8,7))

plt.title('Tweets by Media Type')



ind = np.arange(3)

lst = graph.values



for x, y in zip(ind, lst):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
graph = df[['Tweets', 'Retweets', 'Favourites']].sum()

graph.plot.bar(figsize = (8,7))

plt.title('Tweets, Retweets and Favourites Comparison')



ind = np.arange(3)

lst = graph.values



for x, y in zip(ind, lst):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
#Extract only the data required for analysis.

tweets_df = df[['date', 'Created Date', 'Tweet', 'Retweets', 'Favourites', 'Engagement', 'hour', 'Tweets']]

tweets_df.set_index('date', inplace=True)



tweets_df.head()
#plot graph.

plt.hist(tweets_df['Favourites'], bins=50)

plt.title('Favourites')

plt.show()
df_w = tweets_df.set_index([tweets_df.index.weekday, tweets_df.index])

df_w.sort_index(inplace=True)

df_w.index.names = ['weekday', 'date']

df_sum = df_w.sum(level='weekday')

df_sum = df_sum.drop(columns = 'hour')

df_sum
#plot graph.

df_sum.index = list(calendar.day_abbr)

plt.figure(figsize=(8, 7))

plt.bar(df_sum.index, df_sum['Favourites'])

plt.title('Favourites by day of the week')    



for x, y in zip(df_sum.index, df_sum['Favourites']):

    plt.text(x, y, y, ha='center', va='bottom')

      

plt.show()
#plot graph.

df_sum.index = list(calendar.day_abbr)

plt.figure(figsize=(8, 7))

plt.bar(df_sum.index, df_sum['Tweets'])

plt.title('Tweets by day of the week')



for x, y in zip(df_sum.index, df_sum['Tweets']):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
#Create favourites and time dataframes.

time_df = df[['Favourites','hour']]



#Sort by time.

time_df = time_df.sort_values(by=['hour'], ascending=True)



#Aggregate data by time.

grouped = time_df.groupby('hour')



#Average number of likes per time.

mean = grouped.mean()



#Number of tweets per time.

size = grouped.size()
#Describes a bar graph of the average number of likes at each time.

mean.plot.bar(xlim=[0,24], ylim=[0,7000],figsize=(16,9))

plt.title('Favourites by hour of day')

plt.show()
#Describes a line graph of the number of tweets for each time.

size.plot.bar(xlim=[0,24], ylim=[0,50],figsize=(16,9), label="Tweet count")

plt.title('Tweet count by hour of day')

plt.legend(ncol=1)

plt.show()
df['day_name'] = df['date'].dt.weekday_name;

df_day_hour2 = pd.pivot_table(df[['day_name', 'hour', 'Favourites']], index=['day_name', 'hour'], aggfunc=np.sum, fill_value=0)

df_day_hour3 = df_day_hour2.unstack(level=0)

df_day_hour3 = df_day_hour3.fillna(0)

df_day_hour3 = df_day_hour3.astype(int)

df_day_hour3
morning_hours = []

for hour in range(1, 12):

    detailed_hour = str(hour) + "am"

    morning_hours.append(detailed_hour)

    

afternoon_hours = []

for hour in range(1, 12):

    detailed_hour = str(hour) + "pm"

    afternoon_hours.append(detailed_hour)



day_short_names = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']

detailed_hours = ["12am"] + morning_hours + ["12pm"] + afternoon_hours



sns.set_context("talk")

f, ax = plt.subplots(figsize=(11, 15))

ax = sns.heatmap(df_day_hour3, annot=True, fmt="d", linewidths=.5, ax=ax, xticklabels=day_short_names, yticklabels=detailed_hours, cmap="Reds")

ax.axes.set_title("Heatmap of Favourites Counts by Day and Hour of Day", fontsize=24, y=1.01)

ax.set(xlabel='Day of Week', ylabel='Favourites Counts by Hour of Day');
rank = df[['date', 'Tweet', 'Retweets', 'Favourites', 'Engagement']].sort_values('Engagement', ascending=False)

rank['rank'] = df[['date', 'Tweet', 'Retweets', 'Favourites', 'Engagement']]['Engagement'].rank(ascending=False)

rank['rank'] = rank['rank'].astype(int)

rank = rank.set_index('rank')

rank.head(10)
#Plot Engagement Trend.

df4 = df[['date', 'Engagement']]

df4.set_index('date', inplace=True)

df4 = df4.resample('W').sum() 



df4['Engagement'].plot()

plt.title('Engagement Trend')

plt.legend(ncol=1)

plt.show()
graph = df[['Tweets', 'Engagement']].sum()

graph.plot.bar(figsize = (8,7))

plt.title('Audience Engagement')



ind = np.arange(3)

lst = graph.values



for x, y in zip(ind, lst):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
# Engagement per tweets.

lst[1] / lst[0]
# From https://www.google.com/get/noto/

!wget -q --show-progress https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip

!unzip -p NotoSansCJKjp-hinted.zip NotoSansCJKjp-Regular.otf > NotoSansCJKjp-Regular.otf

!rm NotoSansCJKjp-hinted.zip
text_list = df['Tweet'].to_list()

text = " ".join(text_list)

#text
from janome.tokenizer import Tokenizer

from wordcloud import WordCloud

import matplotlib.pyplot as plt 



full_text = text

 

t = Tokenizer()

tokens = t.tokenize(full_text)

 

word_list=[]

for token in tokens:

    word = token.surface

    partOfSpeech = token.part_of_speech.split(',')[0]

    partOfSpeech2 = token.part_of_speech.split(',')[1]

     

    if partOfSpeech == "名詞":

        if (partOfSpeech2 != "非自立") and (partOfSpeech2 != "代名詞") and (partOfSpeech2 != "数"):

            word_list.append(word)

 

words_wakati=" ".join(word_list)

#print(words_wakati)  

 

stop_words = ['https','co','ため']  

fpath = '/kaggle/working/NotoSansCJKjp-Regular.otf'  # 日本語フォント指定

 

wordcloud = WordCloud(

    font_path=fpath,

    width=900, height=600,   # default width=400, height=200

    background_color="white",   # default=”black”

    stopwords=set(stop_words),

    max_words=200,   # default=200

    min_font_size=4,   #default=4

    collocations = False   #default = True

    ).generate(words_wakati)

 

plt.figure(figsize=(15,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("word_cloud.png")

plt.show()
#Rating.

tweets_df.loc[tweets_df['Favourites'] >= 4000, 'Rating'] = 'A'

tweets_df.loc[(tweets_df['Favourites'] < 4000)  &  (tweets_df['Favourites'] >= 3000), 'Rating'] = 'B'

tweets_df.loc[(tweets_df['Favourites'] < 3000)  &  (tweets_df['Favourites'] >= 2000), 'Rating'] = 'C'

tweets_df.loc[(tweets_df['Favourites'] < 2000)  &  (tweets_df['Favourites'] >= 1000), 'Rating'] = 'D'

tweets_df.loc[tweets_df['Favourites'] < 1000, 'Rating'] = 'E'



#Get the number of characters in each tweet.

tweets_df['TextLength'] = tweets_df.Tweet.str.len()



#Create a list for evaluation.

rank = ['A', 'B', 'C', 'D', 'E']



#Create a dataframe to store the average number of characters per evaluation.

fav_mean_df = pd.DataFrame(index = rank, columns = ['AverageTextLength'])



#Store the average number of characters in the created data frame.

for i in rank:

    df = tweets_df[tweets_df.Rating == i]

    fav_mean_df.loc[[i],['AverageTextLength']] = df['TextLength'].mean()



#Draw graph.

avg = fav_mean_df.plot.bar(figsize=(16,9))

avg.set_title('Average number of characters', fontsize=30)
# Join all tweet.

text_list = tweets_df['Tweet'].to_list()

text = ' '.join(text_list)



# init janome.

char_filters = [UnicodeNormalizeCharFilter(),RegexReplaceCharFilter(r'[-#./:!?)(|]+', '')]

token_nizer = Tokenizer("/kaggle/input/userdic/userdic.csv", udic_enc="utf8")

token_filters = [POSKeepFilter(["名詞"]), POSStopFilter(["名詞,非自立", "名詞,数", "名詞,代名詞", "名詞,接尾"]),LowerCaseFilter(), TokenCountFilter(sorted=True)]



# run janome.

analyzer = Analyzer(char_filters, token_nizer, token_filters)



analyze_list = list(analyzer.analyze(text))

analyze_list = pd.DataFrame(analyze_list, columns = ['word','count'])

analyze_list.head()
#exclude

excludes = ['p', 'w', 'c', 'm', 'httpstco']



analyze_list = analyze_list[~analyze_list.word.isin(excludes)]



#Get top 30.

vc = analyze_list[:30].sort_values('count')



#Create list.

lst = pd.Series.sort_values(vc['count'])



#Draw graph

graph = vc.plot.barh(x='word', y='count', figsize=(10,10))

graph.set_title('Words that appear frequently')



ind = np.arange(30)



# Write a number in a bar graph.

for x, y in zip(ind, lst):

    plt.text(y, x, y, ha='left', va='center')
fig_list = []

p = 0



vc = analyze_list[:30].sort_values('count', ascending=False)



for i, v in vc.iterrows():

  plt.figure(figsize=(9,6))



  #Extract tweets from tweets.

  tweets_df["Extract"] = tweets_df.Tweet.str.contains(v['word'], case=False)



  ind = np.arange(2)

  lst = tweets_df["Extract"].value_counts().tolist()



  plt.subplot(1, 2, 1)



  #Draw graph

  tweets_df["Extract"].value_counts().plot.bar(ind, lst, title=v['word']+'Tweets include / not include')



  # Write a number in a bar graph.

  for x, y in zip(ind, lst):

      plt.text(x, y, y, ha='center', va='bottom')



  #Get average favourites of tweets that do contain word.

  df = tweets_df[tweets_df.Extract == True]

  y1 = int(df["Favourites"].mean())



  #Get average favourites of tweets that do not contain word.

  df = tweets_df[tweets_df.Extract == False]

  y2 = int(df["Favourites"].mean())



  #List each value.

  x = ["include", "not include"]

  y = [y1, y2]



  #Draw graph.

  plt.subplot(1, 2, 2)

  plt.bar(x, y)

  ind = np.arange(2)



  plt.title(v['word']+'Tweets include / not include')



  # Write a number in a bar graph.

  for x, y in zip(ind, y):

    plt.text(x, y, y, ha='center', va='bottom')



  #p = p+1

  plt.show()