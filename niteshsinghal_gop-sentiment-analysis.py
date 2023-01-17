import sklearn

import matplotlib.pyplot as plt

import pandas

from sklearn.cross_validation import train_test_split

import numpy
tweets= pandas.read_csv("../input/Sentiment.csv")

tweets.head()
tweets=tweets[['text','sentiment']]
def transformSentiment(x):

    if x=='Neutral':

        return 0

    elif x=='Negative':

        return -1

    else:

        return 1
tweets['sentiment']=tweets['sentiment'].apply(transformSentiment)
tweets.head()
mood_count=tweets['sentiment'].value_counts()
Index = [1,2,3]

plt.bar(Index,mood_count)

plt.xticks(Index,['Negative','Neutral','Positive'],rotation=45)

plt.ylabel('Mood Count')

plt.xlabel('Mood')

plt.title('Count of Moods')

plt.show()
from wordcloud import WordCloud,STOPWORDS

import re

from nltk.corpus import stopwords
def cleanedWords(raw_tweet):

    raw_tweet=raw_tweet.replace("RT","")

    raw_tweet=re.sub("@\w+:?", "",raw_tweet)

    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet)

    words = letters_only.lower().split()                            

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words

                       if 'http' not in w

                       and w not in ["gopdebate","gopdebates"]

                       and w not in stops]

    return meaningful_words
def getWordCloud(Tweet, sentiment):

    df=Tweet[Tweet['sentiment'] == sentiment]

    words = ' '.join(df['text'])

    cleaned_word = " ".join(cleanedWords(words))

    wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color='black',

                      width=3000,

                      height=2500

                     )

    wordcloud.generate(cleaned_word)

    plt.figure(1,figsize=(12, 12))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
getWordCloud(tweets,-1)
getWordCloud(tweets,0)
getWordCloud(tweets,1)
import nltk
def tweet_to_words(raw_tweet):

    return( " ".join( cleanedWords(raw_tweet) ))
tweets['clean_tweet']=tweets['text'].apply(lambda x: tweet_to_words(x))
tweets[['clean_tweet','text']].head()
train,test = train_test_split(tweets,test_size=0.1,random_state=42)
from sklearn.feature_extraction.text import CountVectorizer

v = CountVectorizer(analyzer = "word")

train_features= v.fit_transform(train['clean_tweet'].values)

test_features=v.transform(test['clean_tweet'].values)
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

    print('Accuracy of '+classifier.__class__.__name__+' is '+str(accuracy))
Index = [1,2,3,4,5,6,7]

plt.bar(Index, Accuracy)

plt.xticks(Index, Model, rotation=90)

plt.ylabel('Accuracy')

plt.xlabel('Model')

plt.title('Accuracies of Models')

plt.show()