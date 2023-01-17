import numpy as np 

import pandas as pd 



import sklearn.feature_extraction.text as text



import spacy 

from wordcloud import WordCloud,STOPWORDS, ImageColorGenerator

nlp=spacy.load("en_core_web_lg")



import textblob

import re



import PIL

from io import BytesIO

import urllib.request

from wordcloud import ImageColorGenerator



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data=pd.read_csv('../input/trump-tweets/trump_tweets.csv', parse_dates = [2])

data
data = data[['date', 'text']]

data.set_index(['date'], inplace = True)

data
def processTweets(tweet):

    '''Cleans the tweet and returns it'''

    tweet = str(tweet)

    tweet = re.sub(r'[^\w\s]','',tweet) # Remove Punctuations

    return tweet



data['text'] = data['text'].apply(processTweets)



data
text = " ".join(text for text in data.text)



stopwords = set(STOPWORDS)



wordcloud = WordCloud(stopwords=stopwords,

                      background_color="white",

                      width = 1920,

                      height = 1080).generate(text)

plt.figure(figsize = (20, 10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
con_mask=np.array(PIL.Image.open('../input/word-cloud-masks/donald-j-trump-1271634_640.png'))



wc = WordCloud(max_words=500, mask=con_mask,width=5000,height=2500,background_color="White",stopwords=STOPWORDS).generate(text)

plt.figure( figsize=(30,15))

plt.imshow(wc)

plt.axis("off")

plt.yticks([])

plt.xticks([])

plt.savefig('./biden.png', dpi=50)

plt.show()