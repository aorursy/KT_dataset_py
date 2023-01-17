# import the libarires 

import numpy as np

import pandas as pd
# load the train dataset

train = pd.read_csv('../input/nlp-getting-started/train.csv')
# check out the train dataset

train.head()
# what is the shape of the dataset

train.shape
# quickly check out the info

train.info()
# first 10 text

train.text[:10]
# check out the distribution of target value

import matplotlib.pyplot as plt 

import seaborn as sns

sns.countplot(train.target)
data  = train.text.tolist()

data[:5]
# lower the values

data = [str(sentence).lower() for sentence in data]

data[:1]
import re



# remove the https/www or http

data = [re.sub(r"https|www|http\S+", "", sent) for sent in data]

data[:50]


data = [re.sub('\S*#|$|@|=|>|!|:\S*\s?', '', sent) for sent in data]

data[:50]
# Substituting multiple spaces with single space

data = [re.sub(r'\s+', ' ', sent) for sent in data]

data[:1]
# remove the digit

data = [re.sub(r"\d", "", sent) for sent in data]

data = [re.sub(r'\^[a-zA-Z0-9]\s]', "", sent) for sent in data]

data[:5]
train['clean_data'] = pd.DataFrame(data)
train.info()
# importing all necessery modules 

from wordcloud import WordCloud, STOPWORDS 
comment_words = ' '

stopwords = set(STOPWORDS) 



for data in train.clean_data: 



    # split the value 

    tokens = data.split() 

    for i in range(len(tokens)): 

        tokens[i] = tokens[i].lower() 

          

    for words in tokens: 

        comment_words = comment_words + words + ' '

  

  

wordcloud = WordCloud(width = 500, height = 500, 

                background_color ='white', 

                stopwords = stopwords, 

                min_font_size = 8).generate(comment_words) 

                  

plt.figure(figsize = (10, 10)) 

plt.imshow(wordcloud) 

plt.show() 