from gensim.models import Word2Vec
from multiprocessing import Pool
import sqlite3 as sql
import numpy as np
import logging
import time
import re

db = '''../input/english-wikipedia-articles-20170820-sqlite/enwiki-20170820.db'''
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def get_query(select, db=db):
    '''
    1. Connects to SQLite database (db)
    2. Executes select statement
    3. Return results and column names
    
    Input: 'select * from analytics limit 2'
    Output: ([(1, 2, 3)], ['col_1', 'col_2', 'col_3'])
    '''
    with sql.connect(db) as conn:
        c = conn.cursor()
        c.execute(select)
        col_names = [str(name[0]).lower() for name in c.description]
    return c.fetchall(), col_names

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
    
def get_section(rowid):
    '''
    1. Construct select statement
    2. Retrieves section_text
    3. Tokenizes section_text
    4. Returns list of tokens

    Input: 100
    Output: ['the','austroasiatic','languages','in',...]
    '''
    select = '''select section_text from articles where rowid=%d''' % rowid
    doc, _ = get_query(select)
    tokens = tokenize(doc[0][0])
    return tokens
       
class Corpus():
    def __init__(self, rowids):
        self.rowids = rowids
        self.len = len(rowids)

    def __iter__(self):
        rowids = np.random.choice(self.rowids, self.len, replace=False)
        with Pool(processes=4) as pool:
            docs = pool.imap_unordered(get_section, rowids)
            for doc in docs:
                yield doc

    def __len__(self):
        return self.len
select = '''select distinct rowid from articles'''
rowids, _ = get_query(select)
rowids = [rowid[0] for rowid in rowids]
start = time.time()
# To keep training time reasonable, let's just look at a random 10K section text sample.
sample_rowids = np.random.choice(rowids, 10000, replace=False)
docs = Corpus(sample_rowids)
word2vec = Word2Vec(docs, min_count=100, size=100)
end = time.time()
print('Time to train word2vec from generator: %0.2fs' % (end - start))
word2vec = Word2Vec.load('../input/english-wikipedia-articles-20170820-models/enwiki_2017_08_20_word2vec.model')
