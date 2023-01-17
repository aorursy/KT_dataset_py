# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import re

import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# Any results you write to the current directory are saved as output.
df= pd.read_csv("../input/Tweets.csv")

df.head()
print("Shape of the dataframe is",df.shape)

print("The number of nulls in each column are \n", df.isna().sum())
print("Percentage null or na values in df")

((df.isnull() | df.isna()).sum() * 100 / df.index.size).round(2)
del df['tweet_coord']

del df['airline_sentiment_gold']

del df['negativereason_gold']

df.head()
print("Total number of tweets for each airline \n ",df.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False))

airlines= ['US Airways','United','American','Southwest','Delta','Virgin America']

plt.figure(1,figsize=(12, 12))

for i in airlines:

    indices= airlines.index(i)

    plt.subplot(2,3,indices+1)

    new_df=df[df['airline']==i]

    count=new_df['airline_sentiment'].value_counts()

    Index = [1,2,3]

    plt.bar(Index,count, color=['red', 'green', 'blue'])

    plt.xticks(Index,['negative','neutral','positive'])

    plt.ylabel('Mood Count')

    plt.xlabel('Mood')

    plt.title('Count of Moods of '+i)
from wordcloud import WordCloud,STOPWORDS
new_df=df[df['airline_sentiment']=='negative']

words = ' '.join(new_df['text'])

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=3000,

                      height=2500

                     ).generate(cleaned_word)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
new_df=df[df['airline_sentiment']=='positive']

words = ' '.join(new_df['text'])

cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and word != 'RT'

                            ])

wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=3000,

                      height=2500

                     ).generate(cleaned_word)

plt.figure(1,figsize=(12, 12))

plt.imshow(wordcloud)

plt.axis('off')

plt.show()




# Calculate highest frequency words in positive tweets

def freq(str): 

  

    # break the string into list of words  

    str = str.split()          

    str2 = [] 

  

    # loop till string values present in list str 

    for i in str:              

  

        # checking for the duplicacy 

        if i not in str2: 

  

            # insert value in str2 

            str2.append(i)  

              

    for i in range(0, len(str2)): 

        if(str.count(str2[i])>50): 

            print('Frequency of', str2[i], 'is :', str.count(str2[i]))

        

print(freq(cleaned_word))
#get the number of negative reasons

df['negativereason'].nunique()



NR_Count=dict(df['negativereason'].value_counts(sort=False))

def NR_Count(Airline):

    if Airline=='All':

        a=df

    else:

        a=df[df['airline']==Airline]

    count=dict(a['negativereason'].value_counts())

    Unique_reason=list(df['negativereason'].unique())

    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']

    Reason_frame=pd.DataFrame({'Reasons':Unique_reason})

    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])

    return Reason_frame

def plot_reason(Airline):

    

    a=NR_Count(Airline)

    count=a['count']

    Index = range(1,(len(a)+1))

    plt.bar(Index,count, color=['red','yellow','blue','green','black','brown','gray','cyan','purple','orange'])

    plt.xticks(Index,a['Reasons'],rotation=90)

    plt.ylabel('Count')

    plt.xlabel('Reason')

    plt.title('Count of Reasons for '+Airline)

    

plot_reason('All')

plt.figure(2,figsize=(13, 13))

for i in airlines:

    indices= airlines.index(i)

    plt.subplot(2,3,indices+1)

    plt.subplots_adjust(hspace=0.9)

    plot_reason(i)
date = df.reset_index()

#convert the Date column to pandas datetime

date.tweet_created = pd.to_datetime(date.tweet_created)

#Reduce the dates in the date column to only the date and no time stamp using the 'dt.date' method

date.tweet_created = date.tweet_created.dt.date

date.tweet_created.head()

df = date

day_df = df.groupby(['tweet_created','airline','airline_sentiment']).size()

# day_df = day_df.reset_index()

day_df
day_df = day_df.loc(axis=0)[:,:,'negative']



#groupby and plot data

ax2 = day_df.groupby(['tweet_created','airline']).sum().unstack().plot(kind = 'bar', color=['red', 'green', 'blue','yellow','purple','orange'], figsize = (15,6), rot = 70)

labels = ['American','Delta','Southwest','US Airways','United','Virgin America']

ax2.legend(labels = labels)

ax2.set_xlabel('Date')

ax2.set_ylabel('Negative Tweets')

plt.show()
def tweet_to_words(tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words ))
df['clean_tweet']=df['text'].apply(lambda x: tweet_to_words(x))
train,test = train_test_split(df,test_size=0.2,random_state=42)
train_clean_tweet=[]

for tweet in train['clean_tweet']:

    train_clean_tweet.append(tweet)

test_clean_tweet=[]

for tweet in test['clean_tweet']:

    test_clean_tweet.append(tweet)
from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(train_clean_tweet)

test_features=v.transform(test_clean_tweet)
Classifiers = [

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=200)]
dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,train['airline_sentiment'])

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,train['airline_sentiment'])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,test['airline_sentiment'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))

    print(classification_report(pred,test['airline_sentiment']))

    cm=confusion_matrix(pred , test['airline_sentiment'])

    plt.figure()

    plot_confusion_matrix(cm,figsize=(12,8), hide_ticks=True,cmap=plt.cm.Reds)

    plt.xticks(range(2), ['Negative', 'Neutral', 'Positive'], fontsize=16,color='black')

    plt.yticks(range(2), ['Negative', 'Neutral', 'Positive'], fontsize=16)

    plt.show()