# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split



import nltk

from nltk.corpus import stopwords

from nltk.classify import SklearnClassifier



from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output
data = pd.read_csv('../input/Sentiment.csv')

print(data.head(2))
#There's too many features, keeping only the 'text' and sentiment'columns

data = data[['text','sentiment']]

data.head()
# Split the dataset into a training and a testing set

train, test = train_test_split(data, test_size=0.1)

print(train.shape)

print(test.shape)



# Removing the neutral sentiments

train = train[train.sentiment !='Neutral']

print(train.sentiment.ravel())

print(set(train.sentiment))
train_pos = train[ train['sentiment']=='Positive']

train_pos = train_pos['text']

train_neg = train[ train['sentiment'] == 'Negative']

train_neg = train_neg['text']



def draw_wordcloud(data, color = 'black'):

    words=' '.join(data)

    cleaned_word = " ".join([word for word in words.split()

                      if 'http' not in word

                          and not word.startswith('@')

                          and not word.startswith('#')

                          and word !='RT'

                      ])

    wordcloud = WordCloud(stopwords=STOPWORDS,

                         background_color=color,

                         width=2500,

                         height=2000

                         ).generate(cleaned_word)

    plt.figure(1, figsize=(13, 13))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    

print('Positive words cloud')

draw_wordcloud(train_pos, 'white')

print('Negtive words cloud')

draw_wordcloud(train_neg, 'gray')
print(train_neg.shape)

print(train_pos.shape)