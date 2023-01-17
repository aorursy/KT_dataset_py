import pandas as pd 
import numpy as np
from os import path
import os
import sys
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from shutil import copyfile
import string
import time
import spacy
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
# packagepath = '../input/vaderSentiment/vaderSentiment/vaderSentiment-master/vaderSentiment'
# sys.path.append(packagepath)
copyfile(src = "../input/vadersentiment/vaderSentiment-master/vaderSentiment/vaderSentiment.py", dst="./vaderSentiment.py")
copyfile(src = "../input/vadersentiment/vaderSentiment-master/vaderSentiment/vader_lexicon.txt", dst="./vader_lexicon.txt")
copyfile(src = "../input/vadersentiment/vaderSentiment-master/vaderSentiment/emoji_utf8_lexicon.txt", dst="./emoji_utf8_lexicon.txt")


# os.system('!cp -r ../input/vadersentiment/vaderSentiment-master/* ./')

from vaderSentiment import SentimentIntensityAnalyzer

# CountVectorizer will help calculate word counts
from sklearn.feature_extraction.text import CountVectorizer

# Import the string dictionary that we'll use to remove punctuation
import string

analyser = SentimentIntensityAnalyzer()


def replace_string(str1, str2):
    temp1 = str1
    temp2 = str2
    temp = temp1.replace(temp2, "")
    return temp

def sentiment_analyzer_scores(sentence, sentiment_type):
    score = analyser.polarity_scores(sentence)
    return score[sentiment_type]
# Import datasets

train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

# The row with index 13133 has NaN text, so remove it from the dataset

train[train['text'].isna()]

train.drop(314, inplace = True)

# Make all the text lowercase - casing doesn't matter when 
# we choose our selected text.
train['text'] = train['text'].apply(lambda x: x.lower())
test['text'] = test['text'].apply(lambda x: x.lower())

# Make training/test split
from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(train, train_size = 0.8, random_state=0)
X_train = X_train.copy()
pos_train = X_train[X_train['sentiment'] == 'positive']
neutral_train = X_train[X_train['sentiment'] == 'neutral']
neg_train = X_train[X_train['sentiment'] == 'negative']
X_train['non_selected'] = X_train.apply(lambda x: replace_string(x['text'],x['selected_text']), axis = 1)

# Use CountVectorizer to get the word counts within each dataset

cv = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english')

X_train_cv = cv.fit_transform(X_train['text'])

X_pos = cv.transform(pos_train['text'])
X_neutral = cv.transform(neutral_train['text'])
X_neg = cv.transform(neg_train['text'])

pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

# Create dictionaries of the words within each sentiment group, where the values are the proportions of tweets that 
# contain those words

pos_words = {}
neutral_words = {}
neg_words = {}
non_words = {}

for k in cv.get_feature_names():
    pos = pos_count_df[k].sum()
    neutral = neutral_count_df[k].sum()
    neg = neg_count_df[k].sum()
    
    pos_words[k] = pos/pos_train.shape[0] 
    neutral_words[k] = neutral/neutral_train.shape[0] 
    neg_words[k] = neg/neg_train.shape[0]


# We need to account for the fact that there will be a lot of words used in tweets of every sentiment.  
# Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other 
# sentiments that use that word.

neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
    
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
    
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

def calculate_selected_text(df_row, tol = 0):
    
    tweet = df_row['text']
    sentiment = df_row['sentiment']
    
    if(sentiment == 'neutral'):
        return tweet
    
    elif(sentiment == 'positive'):
        vader_sent = 'pos'
        dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary
    elif(sentiment == 'negative'):
        vader_sent = 'neg'
        dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary

    #sometimes the vader doesn't recognize any of the words properly according to the tweet's sentiment class, so use original method instead
    utilize_vader = False
    if sentiment_analyzer_scores(tweet, vader_sent) > 0:
        utilize_vader = False
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    vader_sum = 0
    if utilize_vader == True: 
        for i in range(len(subsets)):
            new_vader_sum = sentiment_analyzer_scores(lst[i], 'compound')
            if(new_vader_sum > vader_sum ) or (len(lst[i]) < len(selection_str) and new_vader_sum == vader_sum):
                vader_sum = new_vader_sum
                selection_str = lst[i]
    else:
        for i in range(len(subsets)):
            new_sum = 0 # Sum for the current substring
            # Calculate the sum of weights for each word in the substring
            for p in range(len(lst[i])):
                if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):
                    new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]       
                if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in non_words.keys()):
                    new_sum -= non_word_adj[lst[i][p].translate(str.maketrans('','',string.punctuation))]
            # If the sum is greater than the score, update our current selection
            if(new_sum > score + tol):
                score = new_sum
                selection_str = lst[i]

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)

pd.options.mode.chained_assignment = None

tol = 0.001

X_val['predicted_selection'] = ''

for index, row in X_val.iterrows():
    
    selected_text = calculate_selected_text(row, tol)
    
    X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text

def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

# def jaccard(str1, str2):
#     a = str1.lower().split()
#     b = str2.lower().split()
#     c = set(a).intersection(set(b))
#     return float(len(c)) / (len(a) + len(b) - len(c))
X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)


print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))




pos_tr = train[train['sentiment'] == 'positive']
neutral_tr = train[train['sentiment'] == 'neutral']
neg_tr = train[train['sentiment'] == 'negative']

cv = CountVectorizer(max_df=0.95, min_df=2,
                                     max_features=10000,
                                     stop_words='english')

tfidf = CountVectorizer(max_df=0.95, min_df=2,max_features =10000, stop_words='english')

final_cv = cv.fit_transform(train['text'])

X_pos = cv.transform(pos_tr['text'])
X_neutral = cv.transform(neutral_tr['text'])
X_neg = cv.transform(neg_tr['text'])

pos_final_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())
neutral_final_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())
neg_final_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

pos_words = {}
neutral_words = {}
neg_words = {}

for k in cv.get_feature_names():
    pos = pos_final_count_df[k].sum()
    neutral = neutral_final_count_df[k].sum()
    neg = neg_final_count_df[k].sum()
    
    pos_words[k] = pos/pos_train.shape[0] + sentiment_analyzer_scores(k, 'pos')
    neutral_words[k] = neutral/neutral_train.shape[0]   + sentiment_analyzer_scores(k, 'neu')
    neg_words[k] = neg/neg_train.shape[0]  + sentiment_analyzer_scores(k, 'neg')

  
X_train_cv_2 = tfidf.fit_transform(X_train['non_selected'])
X_non = tfidf.transform(X_train['non_selected'])

non_count_df = pd.DataFrame(X_non.toarray(), columns = tfidf.get_feature_names())

for k in tfidf.get_feature_names():
    non = non_count_df[k].sum()
    non_words[k] = non/X_train['non_selected'].shape[0]
    
neg_words_adj = {}
pos_words_adj = {}
neutral_words_adj = {}

non_word_adj = {}

for key, value in non_words.items():
    non_word_adj[key] = non_words[key]

for key, value in neg_words.items():
    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])
    
for key, value in pos_words.items():
    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])
    
for key, value in neutral_words.items():
    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])

tol = 0.001
def f(selected):
    return " ".join(set(selected.lower().split()))
#  sub.selected_text = sub.selected_text.map(f)

for index, row in test.iterrows():
    selected_text = calculate_selected_text(row, tol)
        
    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text

sample.to_csv('submission.csv', index = False)
