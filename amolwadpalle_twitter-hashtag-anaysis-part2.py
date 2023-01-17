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
train = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/train.csv')

test = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')

# remove the unwanted columns from the train and test

test.drop(['id'], axis = 1, inplace = True)

train.drop(['id'], axis = 1, inplace = True)



test.head()
# wordclod for all the words in train

from wordcloud import WordCloud

import matplotlib.pyplot as plt

txt = " ".join(text for text in test['tweet'])



wordcloud = WordCloud(max_font_size = 100, max_words = 50, background_color = 'orange').generate(txt)



plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()
# lets remove the stopwords

import nltk

import re

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



test = pd.read_csv('/kaggle/input/twitter-sentiment-analysis-hatred-speech/test.csv')

# remove the unwanted columns from the train and test

test.drop(['id'], axis = 1, inplace = True)

stop_words = stopwords.words('english')

stop_words.append('@user')





txt = " ".join(text for text in test['tweet'])



print('before -- ',len(txt))

txt = txt.split()

red_txt = []

for i in (range(len(txt))):

    if txt[i] not in stop_words and len(txt[i])>3:

        red_txt.append(txt[i])

print('After===',len(red_txt))

red_txt = ' '.join(red_txt)

red_txt = red_txt.split()



hashtags = []

for i in red_txt:

    ht = re.findall(r"#(\w+)", i)

    hashtags.append(ht)

hashtags



comments = sum(hashtags,[])

freq = nltk.FreqDist(comments)

freq
df = pd.DataFrame({'tag' : list(freq.keys()),

                  'counts' : list(freq.values())

                  })

top10 = df.nlargest(10, 'counts')

top10
import seaborn as sns

import matplotlib.pyplot as plt



plt.figure(figsize=(16,5))

ax = sns.barplot(data=top10, x= "tag", y = "counts")

ax.set(ylabel = 'counts')

plt.show()
# for a better word cloud

top80 = df.nlargest(25, 'counts')

words = ' '.join(words for words in top80['tag'])

wordcloud = WordCloud(max_font_size = 100, max_words = 50, background_color = 'orange').generate(words)

plt.figure(figsize = (12, 6))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis("off")

plt.show()
