# !pip install -U -q kaggle

# !mkdir -p ~/.kaggle

# !echo '{"username":"sidhsatam","key":"846c7036c3ab10803db8e808150818ec"}' > ~/.kaggle/kaggle.json

# !chmod 600 ~/.kaggle/kaggle.json
# !kaggle competitions download -c tweet-sentiment-extraction
# !unzip /content/train.csv.zip
# !pip install vaderSentiment
import pandas as pd

import nltk

import re 

from nltk import ngrams 

from textblob import TextBlob

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import word_tokenize

import numpy as np
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
train['text'] = train['text'].astype(str) 

test['text'] = test['text'].astype(str) 
analyzer = SentimentIntensityAnalyzer()

pos = []

neg = []

for index, row in train.iterrows():

  senti = row['sentiment']

  if row['text'] == row['selected_text']:

    vs = analyzer.polarity_scores(row['text'])

    if senti=='positive' and vs['compound']>=0.05:

      pos.append(vs['compound'])

    elif senti=='negative' and vs['compound']<=-0.05:

      neg.append(vs['compound'])
min_pos = sum(pos)/len(pos)

max_neg = sum(neg)/len(neg)
def answer(row):

    cleantext = re.sub('http[s]?://\S+', '', row['text'])

    analyzer = SentimentIntensityAnalyzer()

    senti = row['sentiment']

    t_pol = analyzer.polarity_scores(cleantext)['compound']

    if (t_pol>min_pos and senti=='positive') or (t_pol<max_neg and senti=='negative') or senti=='neutral':

        return cleantext

    elif (t_pol>0.05 and senti=='positive') or (t_pol<-0.05 and senti=='negative'):

        best_grams = []

        best_scores = []

        for i in range(1, len(word_tokenize(cleantext)), 2):

            words = word_tokenize(cleantext)

            no_grams = i

            grams = [words[i:i+no_grams] for i in range(len(words)-no_grams+1)]

            gramslist = []

            for gram in grams:

                ngram = ' '.join(gram)

                gramslist.append(ngram)

        

            for gram in gramslist:

                gram_pol = analyzer.polarity_scores(gram)['compound']

                if abs(gram_pol) >= abs(t_pol):

                    best_grams.append(gram)

                    best_scores.append(abs(gram_pol))

        try:

            flag = 0

            return best_grams[np.argmax(best_scores)]

        except:

            flag = 1



    if (t_pol<0.05 and senti=='positive') or (t_pol>-0.05 and senti=='negative') or flag==1:

        words = word_tokenize(cleantext)

        if senti == 'positive':

            scores = []

            for word in words:

                scores.append(analyzer.polarity_scores(word)['compound'])

            

            return words[np.argmax(scores)]

            

        elif senti == 'negative':

            scores = []

            for word in words:

                scores.append(analyzer.polarity_scores(word)['compound'])

            

            return words[np.argmin(scores)]
test['selected_text'] = test.apply(answer, axis=1)
final = test[['textID', 'selected_text']]
final.to_csv('submission.csv', index=False)