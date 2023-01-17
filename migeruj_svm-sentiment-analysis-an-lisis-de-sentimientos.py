# import important libraries
from sklearn import metrics
import numpy as np # linear algebra
import pandas as pd # data processing
import re, string, nltk
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
#import data
data = pd.read_csv("../input/Tweets.csv")

#print head
data.head()
#filter data based on training sentiment confidence
data_clean = data.copy()
data_clean = data_clean[data_clean['airline_sentiment_confidence'] > 0.65]
from sklearn.model_selection import train_test_split

train, test = train_test_split(data_clean, test_size=0.2, random_state=1)
train_tweets = train['text'].values
test_tweets = test['text'].values
train_sentiments = train['airline_sentiment']
test_sentiments = test['airline_sentiment']
#import english stopwords
stopword_list = nltk.corpus.stopwords.words('english') 

def tokenize(text): 
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)
    return tknzr.tokenize(text)
    
def remove_stopwords(text):
    tokens = tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def normalize_corpus(corpus):
    
    normalized_corpus = []
    for index, text in enumerate(corpus):
        text = text.lower()
        text = remove_stopwords(text)
        normalized_corpus.append(text)
    return normalized_corpus
# normalization
norm_train = normalize_corpus(train_tweets)
# feature extraction  
vectorizer = CountVectorizer(ngram_range=(1, 2),tokenizer = tokenize)
train_features = vectorizer.fit_transform(norm_train).astype(float)
# build the model
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(max_iter=7)

svm.fit(train_features, train_sentiments)
# normalize test tweets                        
norm_test = normalize_corpus(test_tweets)
# extract features                                     
test_features = vectorizer.transform(norm_test)
# accuracy on testing
svm.score(test_features, test_sentiments)
#prediect sentiment
predicted_sentiments = svm.predict(test_features)
# print evaluation mesures report
report = metrics.classification_report(y_true=test_sentiments, 
                                           y_pred=predicted_sentiments, 
                                           labels=['positive', 'neutral', 'negative'])
print(report)
datatest = ["flying with @united is always a great experience","I dont like that", "@united very bad experience!", "flying with @united is always a great experience. If you don't lose your luggage"]
newdata = vectorizer.transform(datatest)
svm.predict(newdata)
