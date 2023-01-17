import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import calendar

from datetime import date



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = 1000 # specify 'None' if want to read whole file

# kojima_tweets_en.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows

df1 = pd.read_csv('/kaggle/input/kojima_tweets_en.csv', delimiter=',', nrows = nRowsRead)

df1.dataframeName = 'kojima_tweets_en.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
#Remove date and fraction from time data.

df1['hour'] = pd.to_datetime(df1['Created Date']).dt.hour

df1['date'] = pd.to_datetime(df1['Created Date'])
#Add count.

df1['Tweets'] = 1
#Plot Tweets Trend.

df2 = df1[['date', 'Tweets']]

df2.set_index('date', inplace=True)

df2 = df2.resample('W').sum() 



df2['Tweets'].plot(label="Tweet count")

plt.title('Tweets Trend')

plt.legend(ncol=1)

plt.show()
graph = df1[['No of Hashtags', 'No of User Mentions', 'No of URLS added', 'No of Media added']].sum()

graph.plot.bar(figsize=(8,7))

plt.title('Tweets Content')



for x, y in zip(np.arange(4), graph.values):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
graph = df1[['Media Type', 'Tweets']].groupby('Media Type').sum()

graph = graph['Tweets'].astype(int)

graph.plot.bar(figsize = (8,7))

plt.title('Tweets by Media Type')



ind = np.arange(3)

lst = graph.values



for x, y in zip(ind, lst):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
graph = df1[['Tweets', 'Retweets', 'Favourites']].sum()

graph.plot.bar(figsize = (8,7))

plt.title('Tweets, Retweets and Favourites Comparison')



ind = np.arange(3)

lst = graph.values



for x, y in zip(ind, lst):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
#Extract only the data required for analysis.

tweets_df = df1[['date', 'Created Date', 'Tweet', 'Retweets', 'Favourites', 'Engagement', 'hour', 'Tweets']]

tweets_df.set_index('date', inplace=True)



tweets_df.head()
#plot histgram.

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

time_df = tweets_df[['Favourites','hour']]



#Sort by time.

time_df = time_df.sort_values(by=['hour'], ascending=True)



#Aggregate data by time.

grouped = time_df.groupby('hour')



#Average number of likes per time.

mean = grouped.mean()



#Number of tweets per time.

size = grouped.size()
#Describes a bar graph of the average number of likes at each time.

mean.plot.bar(xlim=[0,24], ylim=[0,16000],figsize=(16,9))

plt.title('Favourites by hour of day')

plt.show()
#Describes a line graph of the number of tweets for each time.

size.plot.bar(xlim=[0,24], ylim=[0,20],figsize=(16,9), label="Tweet count")

plt.title('Tweet count by hour of day')

plt.legend(ncol=1)

plt.show()
df1['day_name'] = df1['date'].dt.weekday_name;

df_day_hour2 = pd.pivot_table(df1[['day_name', 'hour', 'Favourites']], index=['day_name', 'hour'], aggfunc=np.sum, fill_value=0)

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
rank = df1[['date', 'Tweet', 'Retweets', 'Favourites', 'Engagement']].sort_values('Engagement', ascending=False)

rank['rank'] = df1[['date', 'Tweet', 'Retweets', 'Favourites', 'Engagement']]['Engagement'].rank(ascending=False)

rank['rank'] = rank['rank'].astype(int)

rank = rank.set_index('rank')

rank.head(10)
#Plot Engagement Trend.

df4 = df1[['date', 'Engagement']]

df4.set_index('date', inplace=True)

df4 = df4.resample('W').sum() 



df4['Engagement'].plot()

plt.title('Engagement Trend')

plt.legend(ncol=1)

plt.show()
graph = df1[['Tweets', 'Engagement']].sum()

graph.plot.bar(figsize = (8,7))

plt.title('Audience Engagement')



ind = np.arange(3)

lst = graph.values



for x, y in zip(ind, lst):

    plt.text(x, y, y, ha='center', va='bottom')



plt.show()
# Engagement per tweets.

lst[1] / lst[0]
text_list = df1['Tweet'].to_list()

text = " ".join(text_list)

#text
from wordcloud import WordCloud

 

stop_words = ['https','co', 'the', 'of', 'IS', 'to', 'YOUR', 'in', 'HAND', 'you', 'and', 'TOMORROW', 'HANDS', 'with', 'this', 'on', 'デススト', 'デスストでつながれ']  

 

wordcloud = WordCloud(

    width=900, height=600,   # default width=400, height=200

    background_color="white",   # default=”black”

    stopwords=set(stop_words),

    max_words=200,   # default=200

    min_font_size=4,   #default=4

    collocations = False   #default = True

    ).generate(text)

 

plt.figure(figsize=(15,12))

plt.imshow(wordcloud)

plt.axis("off")

plt.savefig("word_cloud.png")

plt.show()