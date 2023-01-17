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
import os

import re

import nltk

import numpy as np

from textblob import TextBlob

from nltk.corpus import stopwords

from string import digits,punctuation 

from nltk.tokenize import word_tokenize

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('stopwords') 

nltk.download('averaged_perceptron_tagger')
import pandas as pd

df_train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

df_test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
df_train = df_train.dropna()

df_train.isna().sum()
# Function to remove the unwanted links from the tweets.

def linkremoval(str):

  text = re.sub(r'^https?:\/\/.*[\r\n]*', '', str, flags=re.MULTILINE)

  return text
def sentiments_in_texts(text):

  text = text.split()

  polarity = dict()

  pos_word_list=[]

  neu_word_list=[]

  neg_word_list=[]



  for word in text:

    if (TextBlob(word).sentiment[0]) > 0:

      pos_word_list.append(word)

    elif (TextBlob(word).sentiment[0]) < 0:

      neg_word_list.append(word)

    else:

      neu_word_list.append(word)



  polarity['positive'] = pos_word_list

  polarity['negative'] = neg_word_list

  polarity['neutral'] = neu_word_list 



  return polarity
def overall_sentiment(str):

  text1 = TextBlob(str)

  if text1.sentiment.polarity>0:

    return 'positive'

  if text1.sentiment.polarity<0:

    return 'negative'

  if text1.sentiment.polarity==0:

    return 'neutral'
def get_selected_text(df):

  textlist = []

  for index, row in df.iterrows():

    word_dict = row['polarity_dict']

    textlist.append(word_dict[row['Overall_Sentiment_using_tweet']])

  return textlist
def jaccard(df1, df2):

  jaccard_scores = []

  for i in range(0,len(df2)):

    a = set(df1[i])

    x = " ".join(df2[i])

    b = set(x)

    c = a.intersection(b)

    jaccard_scores.append(float(len(c)) / (len(a) + len(b) - len(c)))

  return jaccard_scores
df_train['text1'] = df_train['text'].apply(linkremoval)

df_train['Overall_Sentiment_using_tweet'] = df_train['text'].apply(overall_sentiment)

df_train['polarity_dict'] = df_train['text1'].apply(sentiments_in_texts)

df_train['sentiment'] = pd.Categorical(df_train['sentiment'])

df_train['sentiment'].cat.categories

df_train['Overall_Sentiment_using_tweet'] = pd.Categorical(df_train['Overall_Sentiment_using_tweet'])

df_train['Overall_Sentiment_using_tweet'].cat.categories



from sklearn.metrics import classification_report

print(classification_report(df_train["sentiment"],df_train["Overall_Sentiment_using_tweet"]))
train_selected_text_pred = get_selected_text(df_train)
train_selected_text = pd.DataFrame()

for i in train_selected_text_pred:

  txt = " ".join(i)

  txt = pd.Series([txt])

  train_selected_text = pd.concat([train_selected_text,txt],axis=0,ignore_index = True)
train_selected_text.head()
jaccard_scores = jaccard(list(df_train['selected_text']),train_selected_text_pred)
train_jaccard_scores = pd.DataFrame()

for i in jaccard_scores:

  score = pd.Series(str(i))

  train_jaccard_scores = pd.concat([train_jaccard_scores,score],axis=0,ignore_index = True)
train_jaccard_scores.head()
print("Mean Jaccard Scores: ", np.mean(jaccard_scores))
df_test['text1'] = df_test['text'].apply(linkremoval)

df_test['Overall_Sentiment_using_tweet'] = df_test['text'].apply(overall_sentiment)

df_test['polarity_dict'] = df_test['text1'].apply(sentiments_in_texts)

df_test['sentiment'] = pd.Categorical(df_test['sentiment'])

df_test['sentiment'].cat.categories

df_test['Overall_Sentiment_using_tweet'] = pd.Categorical(df_test['Overall_Sentiment_using_tweet'])

df_test['Overall_Sentiment_using_tweet'].cat.categories



from sklearn.metrics import classification_report

print(classification_report(df_test["sentiment"],df_test["Overall_Sentiment_using_tweet"]))
test_selected_text_pred = get_selected_text(df_test)

test_selected_text = pd.DataFrame()

for i in test_selected_text_pred:

  txt = " ".join(i)

  txt = pd.Series([txt])

  test_selected_text = pd.concat([test_selected_text,txt],axis=0,ignore_index = True)
submission = df_test['textID']

submission = pd.concat([submission,test_selected_text[0]],axis=1)

submission = submission.rename(columns={0:'selected_text'})

submission.head()
submission.to_csv('submission.csv',index = False)