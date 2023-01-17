from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data Preprocessing and Feature Engineering
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#Model Selection and Validation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
train_tweets = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
test_tweets = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
train_tweets.head()
test_tweets.head()
train_tweets.info()
test_tweets.info()
def form_sentence(text):
    tweet_blob = TextBlob(text)
    return ' '.join(tweet_blob.words)

print(form_sentence(train_tweets['text'].iloc[10]))
print(train_tweets['text'].iloc[10])
def no_user_alpha(text):
    tweet_list = [ele for ele in text.split() if ele != 'user']
    clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
    clean_s = ' '.join(clean_tokens)
    clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
    return clean_mess
print(no_user_alpha(form_sentence(train_tweets['text'].iloc[10])))
print(train_tweets['text'].iloc[10])
def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
tweet_list = '“But it was the figure you cut as an employee, on an employee’s footing with the girls, in work clothes, and being of that tin-tough, creaking, jazzy bazaar of hardware, glassware, chocolate, chicken-feed, jewelry, drygoods, oilcloth, and song hits—that was the big thing; and even being the Atlases of it, under the floor, hearing how the floor bore up under the ambling weight of hundreds, with the fanning, breathing movie organ next door and the rumble descending from the trolleys on Chicago Avenue—the bloody-rinded Saturday gloom of wind-borne ash, and blackened forms of five-storey buildings rising up to a blind Northern dimness from the Christmas blaze of shops.”'.split()
print(normalization(tweet_list))
