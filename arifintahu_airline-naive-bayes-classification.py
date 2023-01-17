import numpy as np

import pandas as pd



import nltk

from nltk import word_tokenize, pos_tag

from nltk.stem.wordnet import WordNetLemmatizer



from textblob.classifiers import NaiveBayesClassifier as NBC

from textblob import TextBlob



import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        data_path = os.path.join(dirname, filename)

        print(data_path)
# Loading data and picking important features

data = pd.read_csv('/kaggle/input/twitter-airline-sentiment/Tweets.csv')

data = data[['airline_sentiment', 'text']]
data.head()
# Removing word with '@' and 'http'

def preprocess(data):

    result = []

    for row in data.iterrows():

        words = [word.lower() for word in row[1].text.split() if not word.startswith('@')

                                                                and 'http' not in word]

        filtered = ' '.join(words)

        words2 = [word for word in word_tokenize(filtered) if len(word)>=3]

        filtered2 = ' '.join(words2)

        result.append((filtered, row[1].airline_sentiment))

    return result



# Test without preprocess

def extract_data(data):

    result = []

    for row in data.iterrows():

        result.append(( row[1].text, row[1].airline_sentiment))

    return result

data_preprocess = preprocess(data)

data_without_preprocess = extract_data(data)
# Splitting train test data

from sklearn.model_selection import train_test_split



train, test = train_test_split(data_preprocess[:3000], test_size=0.1)

train2, test2 = train_test_split(data_without_preprocess[:3000], test_size=0.1)
from textblob.classifiers import NaiveBayesClassifier as NBC

from textblob import TextBlob
model = NBC(train)

model2 = NBC(train2)
print('Accuracy of model with preprocessing : ', model.accuracy(test))

print('Accuracy of model without preprocessing : ', model2.accuracy(test2))