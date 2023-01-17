%matplotlib inline
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot  as plt
import matplotlib

conn = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql("SELECT * from Sentiment", conn)

#dates processing
df["hour"] = df["tweet_created"].apply(lambda x:pd.to_datetime(x).hour)

# univariate analysis
sns.countplot(y = "candidate" , data = df , order = df["candidate"].value_counts().index)
plt.figure()
pd.Series(df["sentiment"]).value_counts().plot(kind = "pie" , title = "sentiment" , autopct='%.2f')
plt.figure()
sns.countplot(y = "tweet_location" , data = df , order=df["tweet_location"].value_counts()[1:10].index) #top 10 and removing "no location"
plt.figure()
sns.countplot(y = "user_timezone" , data = df , order=df["user_timezone"].value_counts()[1:10].index) #top 10 and removing "no location"
plt.figure()
df["retweet_count"].plot(kind = "hist" , bins = 100 , xlim = (0,300) , figure = plt.figure(5))
plt.figure()
sns.distplot(df["candidate_confidence"] )
plt.figure()
#remove the none of the above for better visualization
sns.countplot(y = "subject_matter" , data = df , order=df["subject_matter"].value_counts()[1:].index)
plt.figure()
sns.countplot(y = "hour" , data = df )


#relation between features
#pd.crosstab(index = df["candidate"] , columns =  df["sentiment"] ).plot(kind = "barh" )
plt.figure()
sns.countplot(y = "candidate" , hue = 'sentiment' , data = df , order = df["candidate"].value_counts().index)
plt.figure()
sent_by_candidates = pd.crosstab(index = df["candidate"] , columns =  df["sentiment"] )
for candidate in sent_by_candidates.index:
    plt.figure()
    sent_by_candidates.loc[candidate].plot(kind = "pie" , title = candidate ,  autopct='%.2f' )

#wordcloud
import re
def cleanword(w):    
    w = re.sub('[^a-zA-Z0-9,]' , ' ' , w)  
    w = re.sub('RT' , ' ' ,w)
    return w

all_text = ""
for ix in df.index:
    all_text += cleanword(df["text"][ix]) + " "


from wordcloud import WordCloud
wordcloud = WordCloud().generate(all_text)
# Open a plot of the generated image.
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
