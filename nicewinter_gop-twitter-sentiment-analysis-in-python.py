from __future__ import division
import pandas as pd
import numpy as np
import requests
import nltk
import string
import re
import os
from os import path
from PIL import Image
from bs4 import BeautifulSoup
from time import sleep
from collections import Counter
from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
%matplotlib inline
raw_data = pd.read_csv('/kaggle/input/first-gop-debate-twitter-sentiment/Sentiment.csv', encoding='utf-8')
tweets = raw_data['text']
labels = raw_data['sentiment'] 
print(tweets.head(2))
print(len(tweets),len(labels))
#Remove all newlines from inside a string
#ref.: https://stackoverflow.com/questions/13298907/remove-all-newlines-from-inside-a-string
clean_tweets = [tweet.replace('\n','').strip() for tweet in tweets]
#To remove all newline characters and then all leading/tailing whitespaces from the string
#Note: strip() only removes the specified characters from the VERY beginning or end of a string. You want to use replace:

#remove the unicodes for the single left and right quote characters - see https://stackoverflow.com/questions/24358361/removing-u2018-and-u2019-character
clean_tweets[:] = [tweet.replace(u'\u2018',"'").replace(u'\u2019',"'") for tweet in clean_tweets] 

#convert abbrevations 
clean_tweets[:] = [tweet.replace('n\'t',' not') for tweet in clean_tweets] #convert n't to  not 

#remove any sub-string containing 'http'
clean_tweets[:] = [re.sub(r"^.*http.*$", '', tweet) for tweet in clean_tweets] 

#remove non-ASCII characters
#see https://stackoverflow.com/questions/20078816/replace-non-ascii-characters-with-a-single-space 
clean_tweets[:] = [re.sub(r'[^\x00-\x7F]+','', tweet) for tweet in clean_tweets] 

#remove tweeter's RT' tags
clean_tweets[:] = [tweet.replace('RT','') for tweet in clean_tweets] 

#make all words lower case
clean_tweets[:] = [tweet.lower() for tweet in clean_tweets] 

clean_tweets[0]
#remove comment mark if you'd like to download for the first time
#nltk.download("stopwords")
string.punctuation
#here I customised the useless words by adding these: 'gop', 'debate', 'gopdebate', 'news' as they are 'neutral' and exist in both positive and negative tweets, won't give any insights; instead, they are likely 'noise' to ML algo.
useless_ones = nltk.corpus.stopwords.words("english") + list(string.punctuation) + ['``', "''",'gop','debate','gopdeb','gopdebate','gopdebates','fox','news','foxnew','foxnews', 'amp']
#useless_ones = nltk.corpus.stopwords.words("english") + list(string.punctuation) 
#remove comment mark if you'd like to download for the first time
#nltk.download("punkt")
#tokenize and clean up the whole set of tweet texts (tokenized and cleaned tweets: tc_tweets)
tc_tweets = []
for tweet in clean_tweets:
    wordlist = [word for word in nltk.word_tokenize(tweet) if word not in useless_ones] #a list of words per tweet
    tc_tweets.append(wordlist)
tc_tweets[0] 
#apply stemming - you can use other stemming algo. 
sno = nltk.stem.SnowballStemmer('english')
tc_tweets_stemmed = []
for words in tc_tweets:
    stemmed_words = [sno.stem(word) for word in words]
    tc_tweets_stemmed.append(stemmed_words)

tc_tweets[:] = tc_tweets_stemmed
#Making a flat list out of list of lists in Python
#ref.: https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
all_words = [item
              for sublist in tc_tweets
              for item in sublist]
len(all_words)
word_counter = Counter(all_words)
most_common_words = word_counter.most_common(10)
most_common_words
sorted_word_counts = sorted(word_counter.values(),reverse=True)
#sorted_word_counts = sorted(list(word_counter.values()), reverse=True)
sorted_word_counts[:10]
plt.loglog(sorted_word_counts)
plt.ylabel("Freq")
plt.xlabel("Word Rank")
plt.title('Word Rank for GOP Twitters')
plt.hist(sorted_word_counts, bins=50, log=True);
label_count=pd.Series(labels).value_counts()
label_count
review_ratio = [opinion/sum(label_count)*100 for opinion in label_count]
print('Sentiment Ratio: ', review_ratio)
Pos_ratio = label_count['Positive']/sum(label_count)*100
print('Positive comments ratio: {0}%'.format(Pos_ratio))
y_pos = range(len(label_count))
#plt.bar(y_pos,label_count,align='center', alpha=.5)
plt.bar(y_pos,review_ratio,align='center', alpha=.5)
plt.xticks(y_pos,label_count.index)
plt.ylabel('Percentage (100%)')
plt.title('Sentiment Ratio for GOP Twitters')
text_label_pair_list = list(zip(tc_tweets,labels))
text_label_pair_list[0]
#remove those neutral tweets as I am only interested in neg / pos ones
text_label_pair_list[:] = [tuple for tuple in text_label_pair_list if tuple[1]!='Neutral']
#split into train and test set, 90% for training set, 10% reserved for testing and evaluation
train, test = train_test_split(text_label_pair_list, test_size = .1, random_state=7)
train_pos = [tuple for tuple in text_label_pair_list if tuple[1]=='Positive']
train_neg = [tuple for tuple in text_label_pair_list if tuple[1]=='Negative']
#unzip texts
train_pos_texts, _ = list(zip(*train_pos))
train_neg_texts, _ = list(zip(*train_neg))
train_pos_texts_str = ' '.join([word for sublist in train_pos_texts
                                        for word in sublist])
train_neg_texts_str = ' '.join([word for sublist in train_neg_texts
                                        for word in sublist])
def create_wordcloud_with_mask (data):
    # read the mask image
    usa_mask = np.array(Image.open(path.join("/kaggle/input/usamap/usa_map.jpg")))
    wcloud = WordCloud(max_words=1000,
                       mask=usa_mask,
                       stopwords=set(STOPWORDS),
                       background_color='white',
                       contour_width=3,
                       contour_color='steelblue')
    #create word cloud
    wcloud.generate(data)
    #display
    plt.figure(1,figsize=(12, 12))
    plt.imshow(wcloud, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
print("Positive comments in training set")
create_wordcloud_with_mask(train_pos_texts_str)
print("Negative comments in training set")
create_wordcloud_with_mask(train_neg_texts_str)
def build_bow_features(words):
    return {word:True for word in words}
#build a list of tuples (BOW_dict, label) for all tweets
train_bow = [(build_bow_features(tuple[0]), tuple[1]) for tuple in train]
test_bow = [(build_bow_features(tuple[0]), tuple[1]) for tuple in test]
print(len(train_bow),len(test_bow))
sentiment_classifier = NaiveBayesClassifier.train(train_bow)
nltk.classify.util.accuracy(sentiment_classifier, train_bow)*100
nltk.classify.util.accuracy(sentiment_classifier, test_bow)*100
test_comment_dicts, test_labels = list(zip(*test_bow))
preds = [sentiment_classifier.classify(comment_dict) for comment_dict in test_comment_dicts]
pred_vs_observ = pd.DataFrame(np.array([test_labels,preds]).T,columns=['observation','prediction'])
pred_vs_observ.transpose()
#print confusion matrix
print(confusion_matrix(test_labels, preds))
sentiment_classifier.show_most_informative_features(100)
