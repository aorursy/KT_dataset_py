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
# Logging using Wrappers

import logging

log_file = "/kaggle/working/logfile.log"

log_level = logging.DEBUG

logging.basicConfig(level=log_level, filename=log_file, filemode="w+",

                        format="%(asctime)-15s %(levelname)-8s %(message)s")

logger = logging.getLogger("baker_logger")



def wrap(pre, post):

    """ Wrapper """

    def decorate(func):

        """ Decorator """

        def call(*args, **kwargs):

            """ Actual wrapping """

            pre(func)

            result = func(*args, **kwargs)

            post(func)

            return result

        return call

    return decorate



def entering(func):

    """ Pre function logging """

    logger.debug("Entered %s", func.__name__)



def exiting(func):

    """ Post function logging """

    logger.debug("Exited  %s", func.__name__)

    

# Ref:= "https://towardsdatascience.com/using-wrappers-to-log-in-python-ccffe4c46b54"
# Concatenate all files into one



import os

import glob

import pandas as pd

os.chdir("/kaggle/input/all-the-news")



extension = 'csv'

all_filenames = [i for i in glob.glob('*.{}'.format(extension))]



#combine all files in the list

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

#export to csv

combined_csv.to_csv( "/kaggle/working/allthenews.csv", index=False, encoding='utf-8-sig')
os.chdir("/kaggle/working")
data = pd.read_csv('allthenews.csv')

data.sort_values(by=['id'], inplace=True)

data.head()
print("The size of the dataset is :=", data.shape)

data['title'].to_csv("titles.csv",index=False)
# Functions for presprocessing data

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from bs4 import BeautifulSoup

import string



tokenizer = RegexpTokenizer(r'\w+') 



def remove_punctuation(text):

    no_punct = "".join([c for c in text if c not in string.punctuation])

    return no_punct



def remove_stopwords(text):

    words = [w for w in text if w not in stopwords.words('english')]

    return words



def word_lemmatizer(text):

    lemmatizer = WordNetLemmatizer()

    lem_text = [lemmatizer.lemmatize(i) for i in text]

    return lem_text



def word_stemmer(text):

    stemmer = PorterStemmer()

    stem_text = [stemmer.stem(i) for i in text]

    return stem_text



def remove_html(text):

    soup = BeautifulSoup(text,'lxml')

    html_free = soup.get_text()

    return html_free



# Ref:= "https://towardsdatascience.com/nlp-for-beginners-cleaning-preprocessing-text-data-ae8e306bef0f"
# Remove NULL and NAN

data['title'] = data['title'].fillna("")

titles = data['title']



# Preprocessing data

titles = titles.apply(lambda x: remove_punctuation(x))

titles = titles.apply(lambda x: tokenizer.tokenize(x.lower()))

titles = titles.apply(lambda x: remove_stopwords(x))

titles = titles.apply(lambda x: word_lemmatizer(x))

#titles = titles.apply(lambda x: word_stemmer(x))
titles.to_csv('titles_processed.csv', index=False)

titles.head(10)
import gensim



title_list = titles.tolist()

# Training the model with all of the titles

model = gensim.models.Word2Vec (title_list, size=50, window=3, min_count=1, workers=3, sg=1)

#model.train(title_list,total_examples=len(title_list),epochs=10)
model.wv.most_similar('new')
w1 = "fret"

model.wv.most_similar (positive=w1)
model.wv.most_similar('bias')