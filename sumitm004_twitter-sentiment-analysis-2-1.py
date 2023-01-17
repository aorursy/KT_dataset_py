import numpy as np

import pandas as pd

import nltk

import re
# Loading the dataset of tweets



df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
# Information about the dataset



print(df_train.info())

print(df_train.head())
df_test.info()
# Checking Class Distribution



df_train['label'].value_counts()
# Storing the tweets and the labels



tweets = df_train['tweet'].str.lower()

tweets_test = df_test['tweet'].str.lower()

Y = df_train['label']



tweets
# Replacing @handle with the word USER



tweets = tweets.str.replace(r'@[\S]+', 'user')

tweets_test = tweets_test.str.replace(r'@[\S]+', 'user')



# Replacing the Hast tag with the word hASH



tweets = tweets.str.replace(r'#(\S+)','hash')

tweets_test = tweets_test.str.replace(r'#(\S+)','hash')



# Removing the all the Retweets



tweets = tweets.str.replace(r'\brt\b',' ')

tweets_test = tweets_test.str.replace(r'\brt\b',' ')



tweets
df_train['tweet'].str.extractall(r'((www\.[\S]+)|(http?://[\S]+))')
# Replacing the URL or Web Address



tweets = tweets.str.replace(r'((www\.[\S]+)|(http?://[\S]+))','URL')

tweets_test = tweets_test.str.replace(r'((www\.[\S]+)|(http?://[\S]+))','URL')



# Replacing Two or more dots with one



tweets = tweets.str.replace(r'\.{2,}', ' ')

tweets_test = tweets_test.str.replace(r'\.{2,}', ' ')
# Removing all the special Characters



tweets = tweets.str.replace(r'[^\w\d\s]',' ')

tweets_test = tweets_test.str.replace(r'[^\w\d\s]',' ')



# Removing all the non ASCII characters



tweets = tweets.str.replace(r'[^\x00-\x7F]+',' ')

tweets_test = tweets_test.str.replace(r'[^\x00-\x7F]+',' ')



# Removing the leading and trailing Whitespaces



tweets = tweets.str.replace(r'^\s+|\s+?$','')

tweets_test = tweets_test.str.replace(r'^\s+|\s+?$','')



# Replacing multiple Spaces with Single Space



tweets = tweets.str.replace(r'\s+',' ')

tweets_test = tweets_test.str.replace(r'\s+',' ')



tweets
# Removing the Stopwords



from nltk.corpus import stopwords



stop_words = set(stopwords.words('english'))



tweets = tweets.apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))



tweets_test = tweets_test.apply(lambda x: ' '.join(word1 for word1 in x.split() if word1 not in stop_words))
# Removing the words stem using Snowball Stemmer



from nltk.stem import *



SS = SnowballStemmer("english")



tweets = tweets.apply(lambda x: ' '.join(SS.stem(word) for word in x.split()))



tweets_test = tweets_test.apply(lambda x: ' '.join(SS.stem(word1) for word1 in x.split()))



tweets
from nltk.tokenize import word_tokenize



# Creating a Bag of Words



words = []

words_test = []



for text in tweets:

    word = word_tokenize(text)

    for i in word:

        words.append(i)

        

        

for text in tweets_test:

    word1 = word_tokenize(text)

    for j in word1:

        words_test.append(i)
from nltk.probability import FreqDist



words = nltk.FreqDist(words)

words_test = nltk.FreqDist(words_test)



print ("Total Number of words in train set {}".format(len(words)))

print ("First 30 most common words in train set {}".format(words.most_common(30)))
# Choosing the first 5000 words as Features



word_features = list(words.keys())[:5000]
# Finding if a word in the word_features is present in the tweets

def finding_features(tweet):

    text = word_tokenize(tweet)

    features = {}

    for i in word_features:

        features[i] = (i in text)

    return features





# Zipping the Processed tweets with the Labels

tweets_featlab = zip(tweets, Y)
# Calling the finding_feature function for all the tweets

feature_set = [(finding_features(TW) ,label) for (TW,label) in tweets_featlab ]
# Calling the finding_feature function for all the test tweets

feature_test = tuple(finding_features(TW) for (TW) in tweets_test)
seed=1

np.random.seed = seed

np.random.shuffle = feature_set



# Splitting Training and Testing Datasets

from sklearn.model_selection import train_test_split



train, test = train_test_split(feature_set, test_size = 0.25, random_state = seed)



print ('Training Size: {}'.format(len(train)))

print ('Testing Size: {}'.format(len(test)))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
# Model_1 DecisionTreeClassifier



nltk_model1 = SklearnClassifier(DecisionTreeClassifier())

nltk_model1.train(train)



accuracy = nltk.classify.accuracy(nltk_model1, test)*100

print ("Accuracy of Decision tree: {}".format(accuracy))



# Predictions for the test data set

predictions_1 = nltk_model1.classify_many(feature_test)
# Model_2 Stochastic Gradient Descent



nltk_model2 = SklearnClassifier(SGDClassifier(max_iter = 1000))

nltk_model2.train(train)



accuracy = nltk.classify.accuracy(nltk_model2, test)*100

print ("Accuracy of SGD: {}".format(accuracy))
# Model_3 RandomForestClassifier



nltk_model3 = SklearnClassifier(RandomForestClassifier())

nltk_model3.train(train)



accuracy = nltk.classify.accuracy(nltk_model3, test)*100

print ("Accuracy of Random Forest: {}".format(accuracy))
# Model_4 Logistic Regression



nltk_model4 = SklearnClassifier(LogisticRegression())

nltk_model4.train(train)



accuracy = nltk.classify.accuracy(nltk_model4, test)*100

print ("Accuracy of Logistic Regression: {}".format(accuracy))
test_1, label = zip(*feature_set)



# Classification Report and Confusion Matrix for the Models

models = [nltk_model1, nltk_model2, nltk_model3, nltk_model4]

classifiers = ['Decision Tree', 'SGD', 'Random forest', 'Logistic Regression']

i = 0

for model in models:

    predictions = model.classify_many(test_1)

    class_=classifiers[i]

    print ("Report for {}".format(class_))

    print (classification_report(label, predictions))

    pd.DataFrame(confusion_matrix(label, predictions))

    i+=1
# Predictions for the test data set using the Best Model

predictions_x = nltk_model1.classify_many(feature_test)
# Saving the Result as csv



df_test['label'] = predictions_x

df_test.to_csv('Final CSV', columns = ['id','label'], index = False)