import os
import pandas as pd
def read_train():
    train=pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
    train['text']=train['text'].astype(str)
    train['selected_text']=train['selected_text'].astype(str)
    return train

def read_test():
    test=pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
    test['text']=test['text'].astype(str)
    return test
train_df = read_train()
train_df.head()
train_df.sentiment.value_counts()
# importing required modules
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
#nltk.download('words')
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
## Function to find 'selected_text'

sid = SentimentIntensityAnalyzer()
def phrase(text, sentiment):
  sentiment = sentiment[0:3]
  sen_scores = sid.polarity_scores(text)
  best_phrase, max_score = text, sen_scores[sentiment]
  
  for i in range (2,6):
    n_grams = ngrams(nltk.word_tokenize(text), i)           ## generating n-grams
    n_grams = [ ' '.join(grams) for grams in n_grams]
    for ngram in n_grams:
      #print("\n",ngram)
      sen_scores = sid.polarity_scores(ngram)       ## for each n-gram calculate polarity score
      if (sen_scores[sentiment] > max_score):
          max_score = sen_scores[sentiment]       
          best_phrase = ngram                       ## get the n-gram (phrase) that gives highest sentiment score for the given sentiment

  return best_phrase
train_df.dropna(inplace=True)       ## eliminating the missing values from the data
train_df.sentiment.value_counts()
## Data Cleaning

import re, string
def clean_text(text):
    text = str(text).lower()
    #text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    return text
train_df['text'] = train_df['text'].apply(clean_text)     ## apply the cleaning function on the 'text' column
import numpy as np

train_df['predicted_phrase'] = np.vectorize(phrase)(train_df['text'], train_df['sentiment'])      ## get the predicted phrase for each pair of text, sentiment
train_df
# Function to find Jaccard Similarity Score
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
train_df['jaccard_score'] = np.vectorize(jaccard)(train_df['predicted_phrase'],train_df['selected_text'])     ## get the jaccard similarity score for the predicted phrases

train_df
train_df['jaccard_score'].mean(axis=0)      ## Acc = Avg(Jaccard scores) = ~35%
test_df = read_test()
test_df.head()
test_df.dropna(inplace=True)
test_df.sentiment.value_counts()
test_df['text'] = test_df['text'].apply(clean_text)
test_df['selected_text'] = np.vectorize(phrase)(test_df['text'], test_df['sentiment'])

test_df
final_df = pd.DataFrame(test_df.iloc[:, [0,3]])

final_df
final_df.to_csv("submission.csv", index=False)      ## save the final df to csv file
