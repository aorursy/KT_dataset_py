from gensim.matutils import sparse2full 

from gensim.corpora import Dictionary

from gensim.models import TfidfModel

from multiprocessing import Pool

from tqdm import tqdm

import sqlite3 as sql

import pandas as pd

import numpy as np

import logging

import time

import re



db = '../input/english-wikipedia-articles-20170820-sqlite/enwiki-20170820.db'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def get_query(select, db=db):

    '''

    1. Connects to SQLite database (db)

    2. Executes select statement

    3. Return results as a dataframe

    '''

    with sql.connect(db) as conn:

        df = pd.read_sql_query(select, conn)

    df.columns = [str(col).lower() for col in df.columns]

    return df
select = '''select * from articles limit 5'''

df = get_query(select)

df
def get_article_text(article_id):

    '''

    1. Construct select statement

    2. Retrieve all section_texts associated with article_id

    3. Join section_texts into a single string (article_text)

    4. Return article_text

    

    Input: 100

    Output: ['the','austroasiatic','languages','in',...]

    '''

    select = '''select section_text from articles where article_id=%d''' % article_id

    df_article = get_query(select)

    doc = '\n'.join(df_article['section_text'].tolist())

    return doc
article_text = get_article_text(0)

# Just print the first 1000 characters for the sake of brevity

print(article_text[:1000])
def tokenize(text, lower=True):

    '''

    1. Strips apostrophes

    2. Searches for all alpha tokens (exception for underscore)

    3. Return list of tokens



    Input: 'The 3 dogs jumped over Scott's tent!'

    Output: ['the', 'dogs', 'jumped', 'over', 'scotts', 'tent']

    '''

    text = re.sub("'", "", text)

    if lower:

        tokens = re.findall('''[a-z_]+''', text.lower())

    else:

        tokens = re.findall('''[A-Za-z_]''', text)

    return tokens
tokens = tokenize(article_text)

print(tokens[:5])
dictionary = Dictionary([tokens])
def get_article_tokens(article_id):

    '''

    1. Construct select statement

    2. Retrieve all section_texts associated with article_id

    3. Join section_texts into a single string (article_text)

    4. Tokenize article_text

    5. Return list of tokens

    

    Input: 100

    Output: ['the','austroasiatic','languages','in',...]

    '''

    select = '''select section_text from articles where article_id=%d''' % article_id

    df_article = get_query(select)

    doc = '\n'.join(df_article['section_text'].tolist())

    tokens = tokenize(doc)

    return tokens
# First, we need to grab all article_ids from the database

select = '''select distinct article_id from articles'''

article_ids = get_query(select)

article_ids = article_ids['article_id'].tolist()
start = time.time()

# Grab a random sample of 10K articles and read into memory

sample_ids = np.random.choice(article_ids, size=10000, replace=False)

docs = []

for sample_id in tqdm(sample_ids):

    docs.append(get_article_tokens(sample_id))

# Train dictionary

dictionary = Dictionary(docs)

end = time.time()

print('Time to train dictionary from in-memory sample: %0.2fs' % (end - start))
start = time.time()

# Grab a random sample of 10K articles and set up a generator

sample_ids = np.random.choice(article_ids, size=10000, replace=False)

docs = (get_article_tokens(sample_id) for sample_id in sample_ids)

# Train dictionary

dictionary = Dictionary(docs)

end = time.time()

print('Time to train dictionary from generator: %0.2fs' % (end - start))
class Corpus():

    def __init__(self, article_ids, db):

        self.article_ids = article_ids

        self.db = db

        self.len = len(article_ids)



    def __iter__(self):

        article_ids_shuffled = np.random.choice(self.article_ids, self.len, replace=False)

        with sql.connect(db) as conn:

            for article_id in article_ids_shuffled:

                select = '''select section_text from articles where article_id=%d''' % article_id

                df_article = self.get_query(select, conn)

                doc = '\n'.join(df_article['section_text'].tolist())

                tokens = self.tokenize(doc)

                yield tokens



    def __len__(self):

        return self.len

        

    def get_query(self, select, conn):

        df = pd.read_sql_query(select, conn)

        df.columns = [str(col).lower() for col in df.columns]

        return df

        

    def tokenize(self, text, lower=True):

        text = re.sub("'", "", text)

        if lower:

            tokens = re.findall('''[a-z_]+''', text.lower())

        else:

            tokens = re.findall('''[A-Za-z_]''', text)

        return tokens
docs = Corpus(article_ids[:10000], db)

dictionary = Dictionary(docs)
dictionary = Dictionary.load('../input/english-wikipedia-articles-20170820-models/enwiki_2017_08_20.dict')