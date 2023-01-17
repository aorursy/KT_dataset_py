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

from textblob import TextBlob

import re

import nltk

from nltk.tokenize import word_tokenize, RegexpTokenizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
train.head()
def get_word_pairs(words):

    new_lis = []

    c = 0

    for i in range(1,len(words)):

        pair_lis = []

        pair_lis.append(words[c])

        pair_lis.append(words[c+1])

        pair = " ".join(pair_lis)

        new_lis.append(pair)

        c+=1

    return new_lis



def get_trigrams(words):

    new_lis = []

    c = 0

    for i in range(1,len(words)-1):

        pair_lis = []

        pair_lis.append(words[c])

        pair_lis.append(words[c+1])

        pair_lis.append(words[c+2])

        pair = " ".join(pair_lis)

        new_lis.append(pair)

        c+=1

    return new_lis
def preprocess(text):

#     # Remove link,user and special characters

#     clean_text = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

#     text = re.sub(clean_text, ' ', str(text)).strip()

    text = text.split()

    return text
train["clean_text"] = train["text"].apply(preprocess)
train["bigrams"] = train["clean_text"].apply(get_word_pairs)

train["trigrams"] = train["clean_text"].apply(get_trigrams)
train.head()
def senti(x):

    scores = []

    for i in x:

        sid = SentimentIntensityAnalyzer()

        text = str(i)

        s = sid.polarity_scores(text)

        scores.append(s["compound"])

    return scores   
train["bigram_scores"] = train["bigrams"].apply(senti)

train["trigram_scores"] = train["trigrams"].apply(senti)
train.head()
def sentence_sentiment(text):

    sid = SentimentIntensityAnalyzer()

    text = str(text)

    score = sid.polarity_scores(text)

    if score['compound'] > 0.05:

        return "Positive"

    elif score['compound'] < -0.05:

        return "Negative"

    else:

        return "Neutral"
train["text_sentiment"] = train["text"].apply(sentence_sentiment)
train.head()
def pos_words(bis,tris,b_score,t_score):

    bis.reverse()

    tris.reverse()

    b_score.reverse()

    t_score.reverse()

    if max(b_score) >= max(t_score):

        return (bis[b_score.index(max(b_score))])

    elif max(b_score) < max(t_score):

        return (tris[t_score.index(max(t_score))])
def neg_words(bis,tris,b_score,t_score):

    bis.reverse()

    tris.reverse()

    b_score.reverse()

    t_score.reverse()

    if min(b_score) <= min(t_score):

        return (bis[b_score.index(min(b_score))])

    elif min(b_score) > min(t_score):

        return (tris[t_score.index(min(t_score))])
def extract_selected_text(df):

    output = []

    for i,x in df.iterrows():

        text_sentiment = sentence_sentiment(x["text"])

        if text_sentiment == "Neutral":

            output.append(x['text']) 

        elif text_sentiment == "Positive":

            if len(x["bigrams"]) == 0:

                output.append(x["text"])

            elif len(x["trigrams"])== 0:

                output.append(pos_words(x["bigrams"],[-1000],x["bigram_scores"],[-1000]))

            else:

                output.append(pos_words(x["bigrams"],x["trigrams"],x["bigram_scores"],x["trigram_scores"]))

        else:

            if len(x["bigrams"]) == 0:

                output.append(x["text"])

            elif len(x["trigrams"])== 0:

                output.append(neg_words(x["bigrams"],[1000],x["bigram_scores"],[1000]))

            else:

                output.append(neg_words(x["bigrams"],x["trigrams"],x["bigram_scores"],x["trigram_scores"]))

    df["selected_text"] = output

    return df.loc[:,["textID","selected_text"]] 
train.head()
output_df = extract_selected_text(train)
output_df
output_df.to_csv('submission.csv',index = False)