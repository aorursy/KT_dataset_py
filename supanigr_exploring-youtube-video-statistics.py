import numpy as np
import pandas as pd
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import seaborn as sb
import glob
%matplotlib inline
files = [file for file in glob.glob('../input/*.{}'.format('csv'))]
sorted(files)
df_initial = list()
for csv in files:
    df_partial = pd.read_csv(csv)
    df_partial['country'] = csv[9:11] #Adding the new column as "country"
    df_initial.append(df_partial)

df = pd.concat(df_initial)
df.info()
df.head()
df.apply(lambda x: sum(x.isnull()))
#Find out the whole column list, without including description column.
column_list_except_description=[] 
for column in df.columns:
    if column not in ["description"]:
        column_list_except_description.append(column)
print(column_list_except_description)
df.dropna(subset=column_list_except_description, inplace=True)
#Let us check for the null entries in new DataFrame
df.apply(lambda x: sum(x.isnull()))
#Import the Data From Json File To Get Our Categories  
category_from_json={}
with open("../input/US_category_id.json","r") as file:
    data=json.load(file)
    for category in data["items"]:
        category_from_json[category["id"]]=category["snippet"]["title"]#it Stores the category id with category name
category_from_json
#Change the "trending_date" and "publish_time" to proper data format.
df["trending_date"]=pd.to_datetime(df["trending_date"],errors='coerce',format="%y.%d.%m")
df["publish_time"]=pd.to_datetime(df["publish_time"],errors='coerce')
#Create some New columns which will help us to dig more into this data.
df["Trending_Year"]=df["trending_date"].apply(lambda time:time.year).astype(int)
df["Trending_Month"]=df["trending_date"].apply(lambda time:time.month).astype(int)
df["Trending_Day"]=df["trending_date"].apply(lambda time:time.day).astype(int)
df["Trending_Day_of_Week"]=df["trending_date"].apply(lambda time:time.dayofweek).astype(int)
df["publish_Year"]=df["publish_time"].apply(lambda time:time.year).astype(int)
df["publish_Month"]=df["publish_time"].apply(lambda time:time.month).astype(int)
df["publish_Day"]=df["publish_time"].apply(lambda time:time.day).astype(int)
df["publish_Day_of_Week"]=df["publish_time"].apply(lambda time:time.dayofweek).astype(int)
df["Publish_Hour"]=df["publish_time"].apply(lambda time:time.hour).astype(int)
#New Data Frame Created But day of week in numeric format we need to convert it.astype(int)
dmap1 = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}#We're Using this Dictionary to Map our column
df["publish_Day_of_Week"]=df["publish_Day_of_Week"].map(dmap1)
df["Trending_Day_of_Week"]=df["Trending_Day_of_Week"].map(dmap1)
df.head(3)
#Lets change the data-format of following fields to int type, which will be easy to use.
list1=["views likes dislikes comment_count".split()] 
for column in list1:
    df[column]=df[column].astype(int)
#Similarly Convert The Category_id into String,because later we're going to map it with data extracted from json file    
list2=["category_id"] 
for column in list2:
    df[column]=df[column].astype(str)
df["Category"]=df["category_id"].map(category_from_json) #category_from_json{} stores categories from JSON files   
df.info()
column_list = ['views', 'likes', 'dislikes', 'comment_count']
corr_matrix = df[column_list].corr()
corr_matrix
plt.figure(figsize = (16,8))

#Let's verify the correlation of each value
ax = sb.heatmap(df[['views', 'likes', 'dislikes', 'comment_count']].corr(), \
            annot=True, annot_kws={"size": 20}, cmap=cm.coolwarm, linewidths=0.5, linecolor='black')
plt.yticks(rotation=30, fontsize=20) 
plt.xticks(rotation=30, fontsize=20) 
plt.title("\nCorrelation between views, likes, dislikes & comments\n", fontsize=25)
plt.show()
def best_publish_time(list, title):
    plt.style.use('ggplot')
    plt.figure(figsize=(16,8))
    #list3=df1.groupby("Publish_Hour").count()["Category"].plot.bar()
    list_temp = list.plot.bar()
    #list3.set_xticklabels(list3.get_xticklabels(),rotation=30, fontsize=15)
    list_temp.set_xticklabels(list_temp.get_xticklabels(),rotation=30, fontsize=15)
    plt.title(title, fontsize=25)
    plt.xlabel(s="Publish_hour", fontsize=20)
    sb.set_context(font_scale=1)
list = df.groupby("Publish_Hour").count()["Category"]
title = "\nOverall, best Publish Time of Youtube Videos\n"
best_publish_time(list, title)
list = df[df['country'] == 'CA'].groupby("Publish_Hour").count()["Category"]
title = "\nBest Publish Time for Canada\n"
best_publish_time(list, title)
list = df[df['country'] == 'DE'].groupby("Publish_Hour").count()["Category"]
title = "\nBest Publish Time for Germany\n"
best_publish_time(list, title)
list = df[df['country'] == 'FR'].groupby("Publish_Hour").count()["Category"]
title = "\nBest Publish Time for France\n"
best_publish_time(list, title)
list = df[df['country'] == 'GB'].groupby("Publish_Hour").count()["Category"]
title = "\nBest Publish Time for Great Britian\n"
best_publish_time(list, title)
plt.figure(figsize=(16,10))#You can Arrange The Size As Per Requirement
list5=df[df['country'] == 'GB'].groupby(["Category","Publish_Hour"]).count()["video_id"].unstack()
ax = sb.heatmap(list5, cmap=cm.coolwarm, linewidths=0.5, linecolor='black')
plt.yticks(rotation=30, fontsize=20) 
plt.xticks(rotation=30, fontsize=20) 
plt.title("\n5AM mystery of GB :)\n", fontsize=30)
plt.show()
list = df[df['country'] == 'US'].groupby("Publish_Hour").count()["Category"]
title = "\nBest Publish Time for USA\n"
best_publish_time(list, title)
plt.figure(figsize=(16,10))#You can Arrange The Size As Per Requirement
list5=df.groupby(["Category","country"]).count()["video_id"].unstack()
ax = sb.heatmap(list5, annot=True, annot_kws={"size": 20},cmap=cm.coolwarm, linewidths=0.5, linecolor='black')
plt.yticks(rotation=30, fontsize=20) 
plt.xticks(rotation=30, fontsize=20) 
plt.title("\nDifferent category of videos from different Country\n", fontsize=20)
plt.show()
df.groupby(["Category","country"]).count()["video_id"].unstack().plot.barh(figsize=(16,10), stacked=True)
plt.yticks(rotation=30, fontsize=20) 
plt.xticks(rotation=30, fontsize=20) 
plt.title("\nDifferent category of videos from different Country\n", fontsize=20)
plt.legend(handlelength=5, fontsize  = 20)
plt.show()
df.groupby(["country", "Category"]).count()["video_id"].unstack().plot.barh(figsize=(12,10), stacked=True)
plt.yticks(rotation=30, fontsize=20) 
plt.xticks(rotation=30, fontsize=15) 
plt.title("\nDifferent category of videos from different Country\n", fontsize=20)
plt.legend(handlelength=5, fontsize  = 15, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
def get_relation(list, title):
    list.plot.bar(stacked=True, figsize=(12,8))
    plt.yticks(rotation=30, fontsize=25) 
    plt.xticks(rotation=30, fontsize=25) 
    plt.title(title, fontsize=25)
    plt.legend(handlelength=4, fontsize  = 25)
    plt.show()
list = df[["country", "views", "likes", "comment_count", "dislikes"]].groupby("country").sum()
title = "\nViews, likes, Dislikes and comment_count\n"
get_relation(list, title)
list = df[["country", "likes", "comment_count", "dislikes"]].groupby("country").sum()
title = "\nlikes, Dislikes and comment_count\n"
get_relation(list, title)
list = df[["country", "comment_count", "dislikes"]].groupby("country").sum()
title = "\nDislikes and comment_count\n"
get_relation(list, title)
def best_videos(list, title):  
    #df_temp = df[["title","views"]].sort_values(by="views",ascending=True).drop_duplicates("title",keep="last")
    list.sort_values(by="views",ascending=False).set_index("title").head(25).plot.bar(figsize=(16,10))
    plt.yticks(rotation=60, fontsize=25) 
    plt.xticks(rotation=90, fontsize=20) 
    plt.title(title, fontsize=25)
    plt.legend(handlelength=5, fontsize  = 30)
    plt.show()
list = df[["title","views"]].sort_values(by="views",ascending=True).drop_duplicates("title",keep="last")
title = "\nThe best 25 videos viewed on Youtube\n"
best_videos(list, title)
list = df[["title","views"]][df["Category"] == "Entertainment"].sort_values(by="views",ascending=True).\
drop_duplicates("title",keep="last")
title = "\nThe best 25  'Entertaintment'  videos viewed on Youtube\n"
best_videos(list, title)
list = df[["title","views"]][df["Category"] == "Music"].sort_values(by="views",ascending=True).\
drop_duplicates("title",keep="last")
title = "\nThe best 25  'Music'  videos viewed on Youtube\n"
best_videos(list, title)
def trend_plot(country):
    df[df["country"] == country][["video_id", "trending_date"]].groupby('video_id').count().sort_values\
    (by="trending_date",ascending=False).plot.kde(figsize=(12,10))
    plt.yticks(rotation=60, fontsize=25) 
    plt.xticks(rotation=30, fontsize=20) 
    plt.title("\nYoutube trend for "+ country +"\n", fontsize=30)
    plt.legend(handlelength=5, fontsize  = 20)
    plt.show()
#country_list = df.groupby(['country']).count().index
country_list = ["FR", "CA"]
for country in country_list:
    trend_plot(country)
country_list = ["GB", "US"]
for country in country_list:
    trend_plot(country)
def pie_chart(df, title):
    labels = df.groupby(['country']).count().index
    sizes = df.groupby(['country']).count()['title']
    fig, ax = plt.subplots(figsize=(8,8))
    ax.pie(sizes, labels=labels, labeldistance=1.1, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax.axis('equal')
    plt.rcParams['font.size'] = 25
    plt.title(title, fontsize=25)
    plt.show()
title = "\nYoutube upload % per country.."
pie_chart(df, title)
#Lets drop the duplicate entries, i,e trending videos, which have multiple entries.
df_temp = df.reset_index().sort_values('trending_date').drop_duplicates('video_id',keep='last')
title = "\nYoutube upload(Unique) % per country..\n"
pie_chart(df_temp, title)
df_temp = df.groupby(["publish_Month","publish_Day_of_Week"]).count()["video_id"].unstack()
plt.figure(figsize=(16,10))
ax = sb.heatmap(df_temp, annot=True, annot_kws={"size": 20},cmap=cm.coolwarm, linewidths=0.5, linecolor='black')
plt.yticks(rotation=30, fontsize=15) 
plt.xticks(rotation=30, fontsize=15) 
plt.title("\nPublish_month vs Publish_day_of_week\n", fontsize=25)
plt.show()
def bar_plot(x,y,title):
    plt.figure(figsize = (16,10))
    sb.barplot(x = x, y = y )
    plt.title(title , fontsize=25)
    plt.yticks(rotation=30, fontsize=15) 
    plt.xticks(rotation=90, fontsize=15) 
    #plt.xticks(rotation = 90)
    plt.show()
x = df.channel_title.value_counts().head(25).index
y = df.channel_title.value_counts().head(25).values
title = "\nTop 25 Channels\n"
bar_plot(x,y,title)
sort_by_comment = df.sort_values(by ="comment_count" , ascending = False).drop_duplicates('title', keep = 'first')
x = sort_by_comment['title'].head(25)
y = sort_by_comment['comment_count'].head(25)
title = "Top Most 25 commented videos"
bar_plot(x,y,title)
from wordcloud import WordCloud
import nltk
#nltk.download("all")
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re
def get_cleaned_data(tag_words):
    #Removes punctuation,numbers and returns list of words
    cleaned_data_set=[]
    cleaned_tag_words = re.sub('[^A-Za-z]+', ' ', tag_words)
    word_tokens = word_tokenize(cleaned_tag_words)
    filtered_sentence = [w for w in word_tokens if not w in en_stopwords]
    without_single_chr = [word for word in filtered_sentence if len(word) > 2]
    cleaned_data_set = [word for word in without_single_chr if not word.isdigit()]  
    return cleaned_data_set

MAX_N = 1000
#Collect all the related stopwords.
en_stopwords = nltk.corpus.stopwords.words('english')
de_stopwords = nltk.corpus.stopwords.words('german')
fr_stopwords = nltk.corpus.stopwords.words('french')   
en_stopwords.extend(de_stopwords)
en_stopwords.extend(fr_stopwords)

polarities = []
category_list = df['Category'].unique()

for category in category_list: #Collect the tag-words for each category
    tag_words = df[df['Category']==category]['tags'].str.lower().str.cat(sep=' ')
    temp_cleaned_data_set = get_cleaned_data(tag_words)
    #Calculate frequency distribution fo the cleaned data.
    word_dist = nltk.FreqDist(temp_cleaned_data_set)
    word_df = pd.DataFrame(word_dist.most_common(MAX_N),
                    columns=['Word', 'Frequency'])

    compound = .0
    for word in word_df['Word'].head(MAX_N):
        compound += SentimentIntensityAnalyzer().polarity_scores(word)['compound']
    polarities.append(compound)

category_list = pd.DataFrame(category_list)
polarities = pd.DataFrame(polarities)
tags_sentiment = pd.concat([category_list,polarities],axis=1)
tags_sentiment.columns = ['category','polarity']
tags_sentiment=tags_sentiment.sort_values('polarity').reset_index()

plt.figure(figsize=(16,10))
ax = sb.barplot(x=tags_sentiment['polarity'],y=tags_sentiment['category'], data=tags_sentiment)
plt.xlabel("Categories")
plt.ylabel("polarity")
plt.yticks(rotation=30, fontsize=25) 
plt.xticks(rotation=90, fontsize=20) 
plt.title("\nPolarity of Different Categories videos\n", fontsize=25)
plt.show()
def word_cloud(category):
    tag_words = df[df['Category']== category]['tags'].str.lower().str.cat(sep=' ')
    temp_cleaned_data_set = get_cleaned_data(tag_words) #get_cleaned_data() defined above.
    
    #Lets plot the word cloud.
    plt.figure(figsize = (20,15))
    cloud = WordCloud(background_color = "white", max_words = 100,  max_font_size = 50)
    cloud.generate(' '.join(temp_cleaned_data_set))
    plt.imshow(cloud)
    plt.axis('off')
    plt.title("\nWord cloud for " + category + "\n", fontsize=40)
category_list = ["Music", "Entertainment"]
for category in category_list:
    word_cloud(category)
category_list = ["News & Politics", "Sports"]
for category in category_list:
    word_cloud(category)