# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

df.shape
df.head()
df.tail()
title = df.copy()

title = title.dropna(subset=['title'])

title['title'] = title['title'].str.replace('[^a-zA-Z]', ' ', regex=True)

title['title'] = title['title'].str.lower()

title.head()
title.tail()
title['keyword_vaccine'] = title['title'].str.find('vaccine') 

title.head()
included_vaccine = title.loc[title['keyword_vaccine'] != -1]

included_vaccine
shaid = []

for index, row in included_vaccine.iterrows():

    id = str(row['sha']) + ".json"

    shaid.append(id)
import json

import os

datafiles = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        if not (filename==''):

            if(filename in shaid):

                ifile = os.path.join(dirname, filename)

                if ifile.split(".")[-1] == "json":

                    datafiles.append(ifile)
len(datafiles)
ArrBodyText = []

for file in datafiles:

    with open(file,'r')as f:

        doc = json.load(f)

    id = doc['paper_id'] 

    bodytext = ''

    for item in doc['body_text']:

        bodytext = bodytext + item['text']

        

    ArrBodyText.append({id:bodytext})
ArrBodyText[20]
text_split = str(ArrBodyText[20]).split()

len(text_split)
#Identify common words

freq = pd.Series(' '.join(text_split).split()).value_counts()[:20]

freq
#Identify uncommon words

freq1 =  pd.Series(' '.join(text_split).split()).value_counts()[-20:]

freq1
from nltk.stem.porter import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

stem = PorterStemmer()

word = "inversely"

print("stemming:",stem.stem(word))

print("lemmatization:", lem.lemmatize(word, "v"))
# Libraries for text preprocessing

import re

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.tokenize import RegexpTokenizer

#nltk.download('wordnet') 

from nltk.stem.wordnet import WordNetLemmatizer
stop_words = set(stopwords.words("english"))



new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","et",'al']

stop_words = stop_words.union(new_words)
corpus = []

for i in range(0, 4548):

    #Remove punctuations

    text = re.sub('[^a-zA-Z]', ' ', text_split[i])

    

    #Convert to lowercase

    text = text.lower()

    

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    

    # remove special characters and digits

    text=re.sub("(\\d|\\W)+"," ",text)

    

    ##Convert to list from string

    text = text.split()

    

    ##Stemming

    ps=PorterStemmer()

    #Lemmatisation

    lem = WordNetLemmatizer()

    text = [lem.lemmatize(word) for word in text if not word in  

            stop_words] 

    text = " ".join(text)

    corpus.append(text)
#View corpus item

corpus[222]
#Word cloud

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

#matplotlib inline

wordcloud = WordCloud(

                          background_color='white',

                          stopwords=stop_words,

                          max_words=100,

                          max_font_size=90, 

                          random_state=62

                         ).generate(str(corpus))

print(wordcloud)

fig = plt.figure(1)

plt.imshow(wordcloud)

plt.axis('off')

plt.show()

fig.savefig("word1.png", dpi=900)