# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import spacy

from spacy import displacy

import datetime as dt

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/wallstreetbets-comments/wallstreetbets_comments.csv')
df.shape
df.head()
df.groupby(['subreddit','subreddit_id']).count()
# since all the rows has subreddit = 'wallstreetbets' and subreddit_id = 't5_2th52', we can drop these two columns

df.drop(columns = ['subreddit','subreddit_id'], inplace = True)
# def get_date(created):

#     return dt.datetime.fromtimestamp(created)



# def get_year_month_hour(df,created):

#     _timestamp = df[created].apply(get_date)

#     df = df.assign(timestamp = _timestamp)

#     df[created[:-4] + '_year'] = pd.DatetimeIndex(df['timestamp']).year

#     df[created[:-4] + '_month'] = pd.DatetimeIndex(df['timestamp']).month

#     df[created[:-4] + '_hour'] = pd.DatetimeIndex(df['timestamp']).hour

#     return df

# df = get_year_month_hour(df,'created_utc')

# df = df.set_index('timestamp')
# df_original = df.copy()
# groupby year barplot

year_groups = df.groupby([df['created_year'], df['created_month']])['created_utc'].count()

year_groups.plot(kind = 'bar',figsize = (25,5))

# plt.xticks(rotation=0)

plt.show()
nyse = pd.read_csv('/kaggle/input/campany-names/nyse.csv')

nasdaq = pd.read_csv('/kaggle/input/campany-names/nasdaq.csv')



tickers = (nyse.Symbol.append(nasdaq.Symbol))



# get the first word of the company's full name

nyse_companies = nyse.Name.str.extract('(?:^|(?:[.!?]\s))(\w+)').drop_duplicates()

nasdaq_companies = nasdaq.Name.str.extract('(?:^|(?:[.!?]\s))(\w+)').drop_duplicates()



companies = nyse_companies.append(nasdaq_companies)[0]
df.body.isna().sum()
# drop comments with body nan

df.dropna(subset = ['body'], inplace = True)
df.shape
# take 20 percent of the data

df = df.sample(frac = .2)

df.shape
import nltk

from gensim.models import Word2Vec

from gensim.models import KeyedVectors

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import PorterStemmer

from nltk.corpus import stopwords 



import itertools

from collections import Counter

from collections import OrderedDict

import operator
# comments_body = df.body



#remove punctuation. Will have to ignore punctuation

tokenizer = nltk.RegexpTokenizer(r"\w+")

df['body'] = df['body'].apply(tokenizer.tokenize)





#removing stop words

stop_words = set(stopwords.words('english'))

stop_words.add('com')

stop_words.add('http')

stop_words.add('www')

stop_words.add('amp')

stop_words.update([str(x) for x in np.arange(10)])

#Blacklisted words (WSB slang from WSB Tickerbot)

stop_words.update([

      "YOLO", "TOS", "CEO", "CFO", "CTO", "DD", "BTFD", "WSB", "OK", "RH",

      "KYS", "FD", "TYS", "US", "USA", "IT", "ATH", "RIP", "BMW", "GDP",

      "OTM", "ATM", "ITM", "IMO", "LOL", "DOJ", "BE", "PR", "PC", "ICE",

      "TYS", "ISIS", "PRAY", "PT", "FBI", "SEC", "GOD", "NOT", "POS", "COD",

      "AYYMD", "FOMO", "TL;DR", "EDIT", "STILL", "LGMA", "WTF", "RAW", "PM",

      "LMAO", "LMFAO", "ROFL", "EZ", "RED", "BEZOS", "TICK", "IS", "DOW"

      "AM", "PM", "LPT", "GOAT", "FL", "CA", "IL", "PDFUA", "MACD", "HQ",

      "OP", "DJIA", "PS", "AH", "TL", "DR", "JAN", "FEB", "JUL", "AUG",

      "SEP", "SEPT", "OCT", "NOV", "DEC", "FDA", "IV", "ER", "IPO", "RISE"

      "IPA", "URL", "MILF", "BUT", "SSN", "FIFA", "USD", "CPU", "AT",

      "GG", "ELON"

   ])



# comments_body = df.body.apply(lambda x: [i for i in x.split(' ') if not i in stop_words])
df['body'] = df.body.apply(lambda x: [i for i in x if not i in stop_words])
def most_common_words(series):

    lst_words = list(itertools.chain.from_iterable(series))

    dict_words = OrderedDict(sorted(Counter(lst_words).items(), key=operator.itemgetter(1), reverse=True))

    return dict_words
top_body = most_common_words(df.body)

english_words = set(nltk.corpus.words.words())



def top_ticker(top_body,ticker_company):

    

    dic = top_body.copy()            

        

    unwanted_tickers = set(dic.keys()) - set(ticker_company)

    for unwanted_key in unwanted_tickers: del dic[unwanted_key]  

        

    return dic



top_body_ticker = top_ticker(top_body,tickers)

top_body_companies = top_ticker(top_body,companies)
to_body_ticker_copy = top_body_ticker.copy()
# print top ten tickers in comments

top_body_ticker = to_body_ticker_copy.copy()

for i in range(10):

    print(top_body_ticker.popitem(last = False))
def mentions_overtime(ticker):

    df_ticker = df[df.body.apply(lambda x: ticker in x)]

    # groupby hour plot

    df_ticker_hour_groups = df_ticker.groupby(['created_year', 'created_month','created_day','created_hour'])['created_utc'].count()

    df_ticker_hour_groups.plot(figsize = (25,5))

    plt.title('Mentions of '+ ticker + ' overtime')

    locs, labels = plt.xticks()

    plt.show()
mentions_overtime('AMD')
mentions_overtime('TSLA')
mentions_overtime('MSFT')
mentions_overtime('MU')
mentions_overtime('AAPL')
mentions_overtime('AMZN')
mentions_overtime('NVDA')
nlp = spacy.load('en',

                    disable=['parser',

                             'tagger',

                             'textcat'])
from tqdm import tqdm_notebook

df.body = df.body.apply(lambda x: ' '.join(x))
# extract named entities

frames = []

for i in tqdm_notebook(range(20000)):

    doc = df.body.iloc[i]

    text_id = df.iloc[i]['id']

    doc = nlp(doc)

    ents = [(e.text,e.start_char, e.end_char, e.label_) for e in doc.ents if len(e.text.strip(' -—')) > 0]

    frame = pd.DataFrame(ents, columns = ['Text','Start','Stop','Type'])

    frame['id'] = text_id

    frames.append(frame)

npf = pd.concat(frames)
npf.Type.value_counts().plot(kind = 'bar',figsize = (25,8))

plt.xticks(rotation=0)

plt.show()
orgs = npf[npf.Type == 'ORG']

orgs.Text.value_counts()[:15].plot(kind='bar',figsize = (25,8))

plt.xticks(rotation=0)

plt.show()
orgs = npf[npf.Type == 'PERSON']

orgs.Text.value_counts()[:15].plot(kind='bar',figsize = (25,8))

plt.xticks(rotation=0)

plt.show()