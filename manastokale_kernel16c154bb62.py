# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import spacy

import re

import string

from nltk.corpus import stopwords

import nltk

from sklearn.ensemble import RandomForestClassifier

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from textblob import TextBlob
nltk.download('vader_lexicon')
stopword = set(stopwords.words('english'))
senti = SentimentIntensityAnalyzer()
dftrain = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

dftest = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
dftrain
dftest
dftrain.dropna(inplace=True)
print(dftrain.isna().sum(),dftrain.shape,dftrain.columns)
#basic cleaning

def clean_text(text):

  #text = x.lower()

  text = re.sub('\[.*?\]', '', text)

  text = re.sub('https?://\S+|www\.\S+', '', text)

  text = re.sub('<.*?>+', '', text)

  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

  text = re.sub('\n', '', text)

  text = re.sub('\w*\d\w*', '', text)

  return text.split(' ')
dftest['cleanedText'] = dftest['text'].apply(clean_text)
print(dftrain.sentiment.value_counts(),'\n')#neutral cases are more causing class imbalance

print(dftest.sentiment.value_counts())
dftest
dftest['SentimentScores'] = None

dftest['Selected Text'] = None
for i in dftest['cleanedText']:

  for x in i:

    if x=='':

      i.remove(x)
dftest['cleanedText'][0]
for i in range(len(dftest['cleanedText'])):

  words = dftest['cleanedText'][i]

  sentiscore = []

  for w in words:

    sentiscore.append(senti.polarity_scores(w)['compound'])

  dftest['SentimentScores'][i] = sentiscore
for i in range(len(dftest['cleanedText'])):

  words = dftest['cleanedText'][i]

  selected_text=[]

  if dftest['sentiment'][i] == 'positive':  

    word = words[np.argmax(dftest['SentimentScores'][i])]

    selected_text.append(word)

  elif dftest['sentiment'][i] == 'negative':  

    word = words[np.argmin(dftest['SentimentScores'][i])]

    selected_text.append(word)

  else:

    selected_text = ' '.join(words)

  dftest['Selected Text'][i] = ''.join(selected_text)
dftest
submission = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
submission['selected_text'] = dftest['Selected Text']
submission
submission.to_csv('submission.csv',index=False)