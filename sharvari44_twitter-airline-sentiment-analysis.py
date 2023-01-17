# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from mlxtend.plotting import plot_confusion_matrix



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,confusion_matrix, f1_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Tweets.csv')

df.head()
df.shape
#Check for NaN values in each column

df.isna().sum()
#Percentage of NaN values

((len(df)-df.count())/len(df))*100
del df['airline_sentiment_gold']

del df['negativereason_gold']

del df['tweet_coord']
df.shape
df.head()
#Sentiment Count 

sentiment_count = df.airline_sentiment.value_counts()

#Airline Review Count

airline_total = df['airline'].value_counts()
index = [1,2,3]

plt.figure(1,figsize=(20,10))

plt.subplot(221)

plt.bar(index,sentiment_count,color=['red','blue','green'])

plt.xticks(index,['negative','neutral','positive'],rotation=0)

plt.xlabel('Sentiment Type')

plt.ylabel('Sentiment Count')

plt.title('Count of Type of Sentiment')

Index=[1,2,3,4,5,6]

my_colors = 'rgbkym'

plt.subplot(222)

plt.bar(Index,airline_total,color=my_colors)

plt.xticks(Index,['United','US Airways','American','Southwest','Delta','Virgin America'],rotation=90)

plt.xlabel('Airline')

plt.ylabel('Review Count')

plt.title('Airline Review Count')
airline_count = df.groupby('airline')['airline_sentiment'].value_counts()
def plot_sentiment_airline(airline):

    df_airline = df[df['airline']==airline]

    count = df_airline['airline_sentiment'].value_counts()

    index = [1,2,3]

    plt.bar(index,count,color=['red','blue','green'])

    plt.xticks(index,['negative','neutral','positive'],rotation=0)

    plt.xlabel('Sentiment Type')

    plt.ylabel('Sentiment Count')

    plt.title('Count of Sentiment Type of '+airline)

airlines = ['US Airways','Virgin America','United','Delta','American','Southwest']

for i in range(len(airlines)):

    plt.figure(1,figsize=(20,12))

    temp = 231+i

    plt.subplot(temp)

    plot_sentiment_airline(airlines[i])
timezone_count = df['user_timezone'].value_counts()

tweet = df.groupby(['airline','airline_sentiment'])['user_timezone'].value_counts()
# Airlines' Negative Sentiment Count by Date

df['tweet_created']=pd.to_datetime(df['tweet_created'])

df['tweet_created'] = df['tweet_created'].dt.date

day = df.groupby(['tweet_created','airline'])['airline_sentiment'].value_counts(sort=True)
date = day.loc(axis=0)[:,:,'negative']

date.groupby(['tweet_created','airline']).sum().unstack().plot(kind='bar',figsize=(15,5))

plt.xlabel('Date')

plt.ylabel('Negative Sentiment Count')

plt.title("Airlines' Negative Sentiment Count by Date")

plt.show()
#Number of Unique Negative Reasons 

df['negativereason'].nunique()
#Negative Reason Count

nr_count = df['negativereason'].value_counts()

nr_dict = dict(df['negativereason'].value_counts())
nr = ["Customer Service Issue","Late Flight","Can't Tell", "Cancelled Flight", "Lost Luggage", "Bad Flight","Flight Booking Problems",         

"Flight Attendant Complaints","longlines","Damaged Luggage"]



def plot_negativereason_count(reason,reason_count):

    index=list(range(10))

    plt.figure(figsize=(15,10))

    plot_colors = 'rgbykcm'

    plt.bar(index,reason_count,color=plot_colors)

    plt.xticks(index,reason,rotation=90)

    plt.xlabel('Negative Reason Type')

    plt.ylabel('Reason Count')

    plt.title('Count of Negative Reasons Type')

plot_negativereason_count(nr,nr_count)
negative_df = df.groupby('airline')['negativereason'].value_counts(ascending=False)
#Negative Reason Count for Airlines

negative_df.groupby(['airline','negativereason']).sum().unstack().plot(kind='bar',figsize=(20,10))

plt.xlabel('Airline')

plt.ylabel('Negative Reason Count')

plt.title("Negative Reason Count for Airlines")

plt.show()
import re

import nltk

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

from mlxtend.plotting import plot_confusion_matrix

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

from sklearn.feature_extraction.text import CountVectorizer
def text_to_words(tweet):

    letters = re.sub("^a-zA-Z"," ",tweet)

    words = letters.lower().split()

    stops = set(stopwords.words("english"))

    meaningful_words = [w for w in words if not w in stops]

    return( " ".join(meaningful_words))
df['sentiment'] = df['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)

df['text_clean'] = df['text'].apply(lambda x:text_to_words(x))
#Train-test split

train,test=train_test_split(df,test_size=0.2,random_state=42)
def clean_tweet(tweet_text):

    res = []

    for tweet in tweet_text:

        res.append(tweet)

    return res

train_clean_text = clean_tweet(train['text_clean'])

test_clean_text = clean_tweet(test['text_clean'])
counter_vectorizer = CountVectorizer(analyzer = "word")

train_features= counter_vectorizer.fit_transform(train_clean_text)

test_features=counter_vectorizer.transform(test_clean_text)
#Classifiers

Classifiers = [

    SVC(kernel="rbf", C=0.025, gamma = 'scale'),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=200),

    GradientBoostingClassifier(),GaussianNB()]
dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for clf in Classifiers:

    try:

        fit = clf.fit(train_features,train['sentiment'])

        pred = fit.predict(test_features)

    except Exception:

        fit = clf.fit(dense_features,train['sentiment'])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,test['sentiment'])

    Accuracy.append(accuracy)

    Model.append(clf.__class__.__name__)

    print('Accuracy of '+clf.__class__.__name__+' is '+str(accuracy)) 

    print(classification_report(pred,test['sentiment']))

    

    #Confusion Matrix

    cm = confusion_matrix(pred,test['sentiment'])

    plt.figure()

    plot_confusion_matrix(cm,cmap=plt.cm.Blues)

    plt.xticks(range(2), ['Negative', 'Positive'],color='black')

    plt.yticks(range(2), ['Negative', 'Positive'])

    plt.xlabel('Predicted Label')

    plt.ylabel('True Label')

    plt.show()
index=list(range(len(Classifiers)))

plt.bar(index,Accuracy,color='rgbyk')

plt.xticks(index,Model,rotation=90)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Classifier Accuracies')