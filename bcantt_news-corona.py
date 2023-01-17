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
import numpy as np

import pandas as pd

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import matplotlib.pyplot as plt

import warnings

import nltk

import re

from nltk.probability import FreqDist

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from unidecode import unidecode

import gensim 

from gensim.models import Word2Vec 

from gensim import corpora, models, similarities

import jieba

import spacy

nlp = spacy.load('en')



warnings.filterwarnings("ignore")
data = pd.read_csv('/kaggle/input/cbc-news-coronavirus-articles-march-26/news.csv')
data
title = ' '.join(data.title.values)



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(title)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(title)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
description = ' '.join(data.description.values)



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(description)



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(description)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
text = ' '.join(data.text.values)

tokenized_word=word_tokenize(text)



stop_words=set(stopwords.words("english"))

filtered_sent=[]

for w in tokenized_word:

    if w not in stop_words:

        filtered_sent.append(w)

        

fdist = FreqDist(filtered_sent)
fdist.plot(30,cumulative=False)

plt.show()
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(' '.join(filtered_sent))

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()


def text_to_word_list(text, remove_polish_letters):

    ''' Pre process and convert texts to a list of words 

    method inspired by method from eliorc github repo: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb'''

    text = remove_polish_letters(text)

    text = str(text)

    text = text.lower()



    # Clean the text

    text = re.sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)

    text = re.sub(r"\+", " plus ", text)

    text = re.sub(r",", " ", text)

    text = re.sub(r"\.", " ", text)

    text = re.sub(r"!", " ! ", text)

    text = re.sub(r"\?", " ? ", text)

    text = re.sub(r"'", " ", text)

    text = re.sub(r":", " : ", text)

    text = re.sub(r"\s{2,}", " ", text)



    text = text.split()



    return text



data.text = data.text.apply(lambda x: text_to_word_list(x, unidecode))
data.text = data.text.apply(lambda x: " ".join(x))
model1 = gensim.models.Word2Vec(data, min_count = 1,  

                              size = 100, window = 5) 
data['group'] = np.nan
data.loc[0,'group'] = 1

for index,row in data.iterrows():

    if nlp(row['text']).similarity(nlp(data.loc[index+1,'text'])) >= 0.95:

        data.loc[index+1,'group'] = data.loc[index,'group']

    else:

        for i in range(1,int(max(data.group.values))+1):

            print(max(data.group.values),'max value')

            if data.loc[data['group'] == i,'text'].shape[0] >= 1:

                if nlp(row['text']).similarity(nlp(data.loc[data['group'] == i,'text'].sample(n=1, random_state=1).values[0])) >= 0.95:

                    data.loc[index+1,'group'] = i

                    break

                else:

                    data.loc[index+1,'group'] = data.loc[index,'group'] + 1

            else:

                data.loc[index+1,'group'] = data.loc[index,'group'] + 1



                

                

    print(index)

        

        

    
data.groupby('group').count()['title'].plot.bar()
