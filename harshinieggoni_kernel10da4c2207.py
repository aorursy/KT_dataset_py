import pandas as pd

import numpy as np

from collections import Counter

import nltk

import pandas as pd

#from emoticons import EmoticonDetector

import re as regex

import numpy as np

#import plotly

#from plotly import graph_objs

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from time import time

import matplotlib.pyplot as plt

%matplotlib inline

#plotly.offline.init_notebook_mode()



import seaborn as sns

import plotly

import cufflinks as cf

import re

nltk.download('punkt')
train_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')



train_data.rename(columns={'Category': 'emotion'}, inplace=True)

test_data.rename(columns={'Category': 'Tweet'}, inplace=True)



train_data = train_data[train_data['emotion'] != 'Tweet']

train_data.head()
train_data.info()
test_data.head()
test_data.info()
sns.countplot(x='emotion',data=train_data)
# remove the tweets which contains Not available



train_data = train_data[train_data['Tweet'] != "Not Available"]
def clean_tweets(tweet):

    

    # remove URL

    tweet = re.sub(r"http\S+", "", tweet)

    

    # Remove usernames

    tweet = re.sub(r"@[^\s]+[\s]?",'',tweet)

    

    # remove special characters 

    tweet = re.sub('[^ a-zA-Z0-9]', '', tweet)

    

    # remove Numbers

    tweet = re.sub('[0-9]', '', tweet)

    

    return tweet
# Apply function to Tweet column



train_data['Tweet'] = train_data['Tweet'].apply(clean_tweets)
'''

text = 'text4 http://url.com/bla2/blah2'

re.sub(r"http\S+", "", text)

text = '@ajay dkfhskf dfs'

re.sub(r"@[^\s]+[\s]?",'',text)

re.sub('[^ a-zA-Z0-9]', '', text)

'''
train_data['Tweet'].head()
# Function which directly tokenize the tweet data

from nltk.tokenize import TweetTokenizer



tt = TweetTokenizer()

train_data['Tweet'].apply(tt.tokenize)
from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize



ps = PorterStemmer()
def tokenize(text):

    return word_tokenize(text)



def stemming(words):

    stem_words = []

    for w in words:

        w = ps.stem(w)

        stem_words.append(w)

    

    return stem_words
# apply tokenize function

train_data['text'] = train_data['Tweet'].apply(tokenize)
# apply steming function

train_data['tokenized'] = train_data['text'].apply(stemming)
train_data.head()
words = Counter()

for idx in train_data.index:

    words.update(train_data.loc[idx, "text"])



words.most_common(5)
nltk.download('stopwords')

stopwords=nltk.corpus.stopwords.words("english")
whitelist = ["n't", "not"]

for idx, stop_word in enumerate(stopwords):

    if stop_word not in whitelist:

        del words[stop_word]

words.most_common(5)
def word_list(processed_data):

    #print(processed_data)

    min_occurrences=3 

    max_occurences=500 

    stopwords=nltk.corpus.stopwords.words("english")

    whitelist = ["n't","not"]

    wordlist = []

    

    whitelist = whitelist if whitelist is None else whitelist

    #print(whitelist)

    '''

    import os

    if os.path.isfile("wordlist.csv"):

        word_df = pd.read_csv("wordlist.csv")

        word_df = word_df[word_df["occurrences"] > min_occurrences]

        wordlist = list(word_df.loc[:, "word"])

        #return

    '''

    words = Counter()

    for idx in processed_data.index:

        words.update(processed_data.loc[idx, "text"])



    for idx, stop_word in enumerate(stopwords):

        if stop_word not in whitelist:

            del words[stop_word]

    #print(words)



    word_df = pd.DataFrame(data={"word": [k for k, v in words.most_common() if min_occurrences < v < max_occurences],

                                 "occurrences": [v for k, v in words.most_common() if min_occurrences < v < max_occurences]},

                           columns=["word", "occurrences"])

    #print(word_df)

    word_df.to_csv("wordlist.csv", index_label="idx")

    wordlist = [k for k, v in words.most_common() if min_occurrences < v < max_occurences]

    #print(wordlist)
word_list(train_data)
words = pd.read_csv("wordlist.csv")
import os
wordlist= []

if os.path.isfile("wordlist.csv"):

    word_df = pd.read_csv("wordlist.csv")

    word_df = word_df[word_df["occurrences"] > 3]

    wordlist = list(word_df.loc[:, "word"])



label_column = ["label"]

columns = label_column + list(map(lambda w: w + "_bow",wordlist))

labels = []

rows = []

for idx in train_data.index:

    current_row = []

    

    # add label

    current_label = train_data.loc[idx, "emotion"]

    labels.append(current_label)

    current_row.append(current_label)



    # add bag-of-words

    tokens = set(train_data.loc[idx, "text"])

    for _, word in enumerate(wordlist):

        current_row.append(1 if word in tokens else 0)



    rows.append(current_row)



data_model = pd.DataFrame(rows, columns=columns)

data_labels = pd.Series(labels)





bow = data_model
import random

seed = 777

random.seed(seed)

def test_classifier(X_train, y_train, X_test, y_test, classifier):

    log("")

    log("---------------------------------------------------------")

    log("Testing " + str(type(classifier).__name__))

    now = time()

    list_of_labels = sorted(list(set(y_train)))

    model = classifier.fit(X_train, y_train)

    log("Learing time {0}s".format(time() - now))

    now = time()

    predictions = model.predict(X_test)

    log("Predicting time {0}s".format(time() - now))



    # Calculate Accuracy, Precision, recall

    

    precision = precision_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)

    recall = recall_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)

    accuracy = accuracy_score(y_test, predictions)

    f1 = f1_score(y_test, predictions, average=None, pos_label=None, labels=list_of_labels)

    

    log("=================== Results ===================")

    log("            Negative     Neutral     Positive")

    log("F1       " + str(f1))

    log("Precision" + str(precision))

    log("Recall   " + str(recall))

    log("Accuracy " + str(accuracy))

    log("===============================================")



    return precision, recall, accuracy, f1



def log(x):

    #can be used to write to log file

    print(x)

from sklearn.naive_bayes import BernoulliNB

X_train, X_test, y_train, y_test = train_test_split(bow.iloc[:, 1:], bow['label'], test_size=0.3)

precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())
def cv(classifier, X_train, y_train):

    log("===============================================")

    classifier_name = str(type(classifier).__name__)

    now = time()

    log("Crossvalidating " + classifier_name + "...")

    accuracy = [cross_val_score(classifier, X_train, y_train, cv=8, n_jobs=-1)]

    log("Crosvalidation completed in {0}s".format(time() - now))

    log("Accuracy: " + str(accuracy[0]))

    log("Average accuracy: " + str(np.array(accuracy[0]).mean()))

    log("===============================================")

    return accuracy
train_data = pd.read_csv('data/train.csv')

test_data = pd.read_csv('data/test.csv')



train_data.rename(columns={'Category': 'emotion'}, inplace=True)

test_data.rename(columns={'Category': 'emotion'}, inplace=True)



train_data = train_data[train_data['emotion'] != 'Tweet']

test_data = test_data[test_data['emotion'] != 'Tweet']
def add_extra_feature(df, tweet_column):

    

    # Print Number of Exclamation

    #length_of_excl = (len(re.findall(r'!', string)))

    df['number_of_exclamation'] = tweet_column.apply(lambda x: (len(re.findall(r'!', x))))

    

    # Number of ?

    #length_of_questionmark = (len(re.findall(r'?', string)))

    df['number_of_questionmark'] = tweet_column.apply(lambda x: (len(re.findall(r'[?]', x))))

    

    # Number of #

    df['number_of_hashtag'] = tweet_column.apply(lambda x: (len(re.findall(r'#', x))))

    

    # Number of @

    df['number_of_mention'] = tweet_column.apply(lambda x: (len(re.findall(r'@', x))))

    

    # Number of Quotes

    df['number_of_quotes'] = tweet_column.apply(lambda x: (len(re.findall(r"'", x))))



    # Number if underscore

    df['number_of_underscore'] = tweet_column.apply(lambda x: (len(re.findall(r'_', x))))

    

    

    #print((txt.split(" "), row))

    #print(row.split())
# pass the train_data into add_extra_feature function

add_extra_feature(train_data, train_data["Tweet"])

## Emoticon Detector



class EmoticonDetector:

    emoticons = {}



    def __init__(self, emoticon_file="data/emoticons.txt"):

        from pathlib import Path

        content = Path(emoticon_file).read_text()

        positive = True

        for line in content.split("\n"):

            if "positive" in line.lower():

                positive = True

                continue

            elif "negative" in line.lower():

                positive = False

                continue



            self.emoticons[line] = positive



    def is_positive(self, emoticon):

        if emoticon in self.emoticons:

            return self.emoticons[emoticon]

        return False



    def is_emoticon(self, to_check):

        return to_check in self.emoticons
ed = EmoticonDetector()
processed_data = train_data.copy()



def add_column(column_name, column_content):

    processed_data.loc[:, column_name] = pd.Series(column_content, index=processed_data.index)



def count_by_lambda(expression, word_array):

    return len(list(filter(expression, word_array)))



add_column("splitted_text", map(lambda txt: txt.split(" "), processed_data["Tweet"]))



positive_emo = list(

    map(lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and ed.is_positive(word), txt),

        processed_data["splitted_text"]))

add_column("number_of_positive_emo", positive_emo)



negative_emo = list(map(

    lambda txt: count_by_lambda(lambda word: ed.is_emoticon(word) and not ed.is_positive(word), txt),

    processed_data["splitted_text"]))



add_column("number_of_negative_emo", negative_emo)
train_data['number_of_positive_emo'] = positive_emo

train_data['number_of_negative_emo'] = negative_emo
sns.barplot(x='emotion', y='number_of_mention', data=train_data)

sns.despine()

plt.tight_layout()
sns.barplot(x='emotion', y='number_of_negative_emo', data=train_data)

sns.despine()

plt.tight_layout()
sns.barplot(x='emotion', y='number_of_questionmark', data=train_data)

sns.despine()

plt.tight_layout()
sns.barplot(x='emotion', y='number_of_exclamation', data=train_data)

sns.despine()

plt.tight_layout()
sns.barplot(x='emotion', y='number_of_positive_emo', data=train_data)

sns.despine()

plt.tight_layout()
sns.barplot(x='emotion', y='number_of_hashtag', data=train_data)

sns.despine()

plt.tight_layout()
sns.barplot(x='emotion', y='number_of_underscore', data=train_data)

sns.despine()

plt.tight_layout()
train_data.head()
# apply the clean tweet function

train_data['Tweet'] = train_data['Tweet'].apply(clean_tweets)
## Tokenize data

train_data['text'] = train_data['Tweet'].apply(tokenize)

train_data['tokenized'] = train_data['text'].apply(stemming)
## BAG OF WORDS

wordlist= []

if os.path.isfile("wordlist.csv"):

    word_df = pd.read_csv("wordlist.csv")

    word_df = word_df[word_df["occurrences"] > 3]

    wordlist = list(word_df.loc[:, "word"])



label_column = ["label"]

columns = label_column + list(map(lambda w: w + "_bow",wordlist))

labels = []

rows = []

for idx in train_data.index:

    current_row = []

        # add label

    current_label = train_data.loc[idx, "emotion"]

    labels.append(current_label)

    current_row.append(current_label)



    # add bag-of-words

    tokens = set(train_data.loc[idx, "text"])

    for _, word in enumerate(wordlist):

        current_row.append(1 if word in tokens else 0)



    rows.append(current_row)



data_model = pd.DataFrame(rows, columns=columns)

data_labels = pd.Series(labels)



dat1 = train_data

dat2 = data_model



dat1 = dat1.reset_index(drop=True)

dat2 = dat2.reset_index(drop=True)



data_model = dat1.join(dat2)
train_data.columns
## Drop the columns in data_model

data_model = data_model.drop(columns=['emotion','Tweet','text', 'tokenized','Id'], axis=1)
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(data_model.drop(columns='label',axis=1),data_model['label'] , test_size=0.3)

precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, RandomForestClassifier(random_state=seed,n_estimators=403,n_jobs=-1))

rf_acc = cv(RandomForestClassifier(n_estimators=403,n_jobs=-1, random_state=seed),data_model.drop(columns='label',axis=1), data_model['label'])
from xgboost import XGBClassifier as XGBoostClassifier
X_train, X_test, y_train, y_test = train_test_split(data_model.drop(columns='label',axis=1),data_model['label'] , test_size=0.3)

precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, XGBoostClassifier(seed=seed))
X_train, X_test, y_train, y_test = train_test_split(data_model.drop(columns='label',axis=1),data_model['label'] , test_size=0.3)

precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())
test_data.head()
test_data.columns
# remove the tweets which contains Not available

test_data = test_data.rename(columns={"emotion": "Tweet"})

test_data = test_data[test_data['Tweet'] != "Not Available"]


# Drop null values

test_data = test_data.dropna() 



# add extra features

add_extra_feature(test_data, test_data['Tweet'])



# Clean tweets

test_data['Tweet'] = test_data['Tweet'].apply(clean_tweets)



## Tokenize data

test_data['text'] = test_data['Tweet'].apply(tokenize)

test_data['tokenized'] = test_data['text'].apply(stemming)
# wordlist

word_list(test_data)
## BAG OF WORDS

wordlist= []

if os.path.isfile("wordlist.csv"):

    word_df = pd.read_csv("wordlist.csv")

    word_df = word_df[word_df["occurrences"] > 3]

    wordlist = list(word_df.loc[:, "word"])



label_column = ["label"]

columns = label_column + list(map(lambda w: w + "_bow",wordlist))

labels = []

rows = []

for idx in test_data.index:

    current_row = []

        # add label

    current_label = test_data.loc[idx, "Tweet"]

    labels.append(current_label)

    current_row.append(current_label)



    # add bag-of-words

    tokens = set(test_data.loc[idx, "text"])

    for _, word in enumerate(wordlist):

        current_row.append(1 if word in tokens else 0)



    rows.append(current_row)



data_model = pd.DataFrame(rows, columns=columns)

data_labels = pd.Series(labels)



dat1 = test_data

dat2 = data_model



dat1 = dat1.reset_index(drop=True)

dat2 = dat2.reset_index(drop=True)



data_model = dat1.join(dat2)
test_model = pd.DataFrame()

test_model['original_id'] = data_model['Id']
data_model = data_model.drop(columns=['Tweet','text', 'tokenized','Id'], axis=1)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=403,max_depth=10)
RF.fit(data_model.drop(columns='label',axis=1),data_model['label'])
predictions = RF.predict(data_model.drop(columns='label',axis=1))
results = pd.DataFrame([],columns=["Id","Category"])

results["Id"] = test_model["original_id"].astype("int64")

results["Category"] = predictions

results.to_csv("results_xgb.csv",index=False)