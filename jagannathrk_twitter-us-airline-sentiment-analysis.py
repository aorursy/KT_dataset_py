import sklearn

import matplotlib.pyplot as plt

%matplotlib inline

import pandas

from sklearn.model_selection import train_test_split

import numpy
Tweet= pandas.read_csv("../input/Tweets.csv")

Tweet.head()
(len(Tweet)-Tweet.count())/len(Tweet)
del Tweet['tweet_coord']

del Tweet['airline_sentiment_gold']

del Tweet['negativereason_gold']
Mood_count=Tweet['airline_sentiment'].value_counts()
Index = [1,2,3]

plt.bar(Index,Mood_count)

plt.xticks(Index,['negative','neutral','positive'],rotation=45)

plt.ylabel('Mood Count')

plt.xlabel('Mood')

plt.title('Count of Moods')
Tweet['airline'].value_counts()
def plot_sub_sentiment(Airline):

    df=Tweet[Tweet['airline']==Airline]

    count=df['airline_sentiment'].value_counts()

    Index = [1,2,3]

    plt.bar(Index,count)

    plt.xticks(Index,['negative','neutral','positive'])

    plt.ylabel('Mood Count')

    plt.xlabel('Mood')

    plt.title('Count of Moods of '+Airline)

plt.figure(1,figsize=(12, 12))

plt.subplot(231)

plot_sub_sentiment('US Airways')

plt.subplot(232)

plot_sub_sentiment('United')

plt.subplot(233)

plot_sub_sentiment('American')

plt.subplot(234)

plot_sub_sentiment('Southwest')

plt.subplot(235)

plot_sub_sentiment('Delta')

plt.subplot(236)

plot_sub_sentiment('Virgin America')
NR_Count=dict(Tweet['negativereason'].value_counts(sort=False))
def NR_Count(Airline):

    if Airline=='All':

        df=Tweet

    else:

        df=Tweet[Tweet['airline']==Airline]

    count=dict(df['negativereason'].value_counts())

    Unique_reason=list(Tweet['negativereason'].unique())

    Unique_reason=[x for x in Unique_reason if str(x) != 'nan']

    Reason_frame=pandas.DataFrame({'Reasons':Unique_reason})

    Reason_frame['count']=Reason_frame['Reasons'].apply(lambda x: count[x])

    return Reason_frame
def plot_reason(Airline):

    df=NR_Count(Airline)

    count=df['count']

    Index = range(1,(len(df)+1))

    plt.bar(Index,count)

    plt.xticks(Index,df['Reasons'],rotation=90)

    plt.ylabel('Count')

    plt.xlabel('Reason')

    plt.title('Count of Reasons for '+Airline)
plot_reason('All')
plot_reason('US Airways')
plot_reason('United')
plot_reason('American')
plot_reason('Southwest')
plot_reason('Delta')
plot_reason('Virgin America')
from wordcloud import WordCloud,STOPWORDS
df=Tweet[Tweet['airline_sentiment']=='negative']

words = ' '.join(df['text'])

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
import re

import nltk

from nltk.corpus import stopwords
def tweet_to_words(raw_tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 
def clean_tweet_length(raw_tweet):

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return(len(meaningful_words)) 
Tweet['sentiment']=Tweet['airline_sentiment'].apply(lambda x: 0 if x=='negative' else 1)
Tweet['clean_tweet']=Tweet['text'].apply(lambda x: tweet_to_words(x))

Tweet['Tweet_length']=Tweet['text'].apply(lambda x: clean_tweet_length(x))

train,test = train_test_split(Tweet,test_size=0.2,random_state=42)
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
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score
Classifiers = [

    LogisticRegression(C=0.000000001,solver='liblinear',max_iter=200),

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(n_estimators=200),

    AdaBoostClassifier(),

    GaussianNB()]
dense_features=train_features.toarray()

dense_test= test_features.toarray()

Accuracy=[]

Model=[]

for classifier in Classifiers:

    try:

        fit = classifier.fit(train_features,train['sentiment'])

        pred = fit.predict(test_features)

    except Exception:

        fit = classifier.fit(dense_features,train['sentiment'])

        pred = fit.predict(dense_test)

    accuracy = accuracy_score(pred,test['sentiment'])

    Accuracy.append(accuracy)

    Model.append(classifier.__class__.__name__)

    print('Accuracy of '+classifier.__class__.__name__+'is '+str(accuracy))    
Index = [1,2,3,4,5,6,7]

plt.bar(Index,Accuracy)

plt.xticks(Index, Model,rotation=45)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Accuracies of Models')