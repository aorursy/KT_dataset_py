# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

training = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv",encoding = 'latin1')

columns = ['sentiment','id','date','query_string','user','text']
df = pd.read_csv("../input/sentiment140/training.1600000.processed.noemoticon.csv",encoding = 'latin-1',header = None,names=columns)
df['text'] = df['text'].str.encode('utf8')
df['text'] = df['text'].str.decode('utf8')

df['sentiment'].value_counts()
df[df['sentiment'] == 0]['sentiment'] # all the negative tweets
df[df['sentiment'] == 4]['sentiment'] # all the positive tweets
df
df.drop(columns = ['id','date','query_string','user'],inplace = True)
df.head()
df[df['sentiment'] == 0].index
df[df['sentiment'] == 4].index
df['sentiment'] = df['sentiment'].map({0:0,4:1})
# lets look at length of each string in the dataset

df['text_length'] = [len(words) for words in df['text']]
df
from pprint import pprint

data_dictonary = {

    'sentiment':{

        'type':df['sentiment'].dtype,

        'description':'sentiment_class 0 : negative, 1:positive'

    },

    'text':{

        'type':df['text'].dtype,

        'description':'individual tweet text'

        

    },

    'text_length':{

        'type':df['text_length'].dtype,

        'deescription':'This is the length of the text before cleaning'

    },

    'shape_of_the_dataset':df.shape

    

}
pprint(data_dictonary)
# lets find out overall distribution of text length of each text

plt.figure(figsize = (15,5))

sns.boxplot(df['text_length'])

# from the plot we observe that there are many outliers but most of the tweets are in the range 50 to 100
# lets take a detailed view at those outlier tweets

df[df['text_length'] > 140].head(4)
df['text'][279] # we observe that there are various HTML tags and encodings in the data lets use beautifulSoup to convert 

# this into readable text
from bs4 import BeautifulSoup

BeautifulSoup(df['text'][279],'lxml').get_text()

# as we can observe that the data is converted into english
df['text'][343]
import re

re.sub(r'@[A-Za-z0-9]+','',df['text'][343]) # in this case we are using regular expressions

# this regular expression signifies a @ symbol followed by any number of characters numbers capital or small

df['text'][0]
re.sub('https?://[A-Za-z0-9./]+','',df['text'][0]) # this regex can be used to remove hyper links
df['text'][226]
re.sub(r'ï¿½','?',df['text'][226])
#the text associated with hashtag is quite important hence we only remove the hashtags in the text and keep the preceding text intact

df['text'][175]
re.sub("[^a-zA-Z]"," ",df['text'][175]) # as we can see by using this regular expression we removed the tweet where character @ and character # is

# substituted by zero
from nltk.tokenize import WordPunctTokenizer

# with help of this tokenizer we can tokenize at space and word punctuationns

tokenizer = WordPunctTokenizer()

regex_at_hyperlink = r"@[A-Za-z)-9]|https?://[A-Za-z0-9./]+"

def data_cleaning(text):

    soup = BeautifulSoup(text,'lxml')

    clean_soup = soup.get_text()

    at_and_hyperlink_removed = re.sub(regex_at_hyperlink,'',clean_soup)

    remove_latin_tokens = re.sub(r'ï¿½','?',at_and_hyperlink_removed)

    hashtag_removed = re.sub("[^a-zA-Z]"," ",remove_latin_tokens) # this process may create unnecessary white spaces so we remove use strip function

    lower_case = hashtag_removed.lower()

    words = tokenizer.tokenize(lower_case)

    return (" ".join(words)).strip()

    
# time to check this function

df['text'][:100]
df['text'][:100].apply(lambda x : data_cleaning(x)) # this cleaner works perfectly
#applying on whole data

clean_tweets = df['text'].apply(lambda x:data_cleaning(x))
clean_tweets.to_csv('clean_tweet.csv',encoding='utf-8')
csv = 'clean_tweet.csv'

my_df = pd.read_csv(csv)

my_df.head()