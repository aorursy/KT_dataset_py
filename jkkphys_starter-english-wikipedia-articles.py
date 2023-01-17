import sqlite3 as sql
import pandas as pd
import re

db = '../input/enwiki-20170820.db'
def get_query(select, db=db):
    '''Executes a select statement and returns results and column/field names.'''
    with sql.connect(db) as conn:
        c = conn.cursor()
        c.execute(select)
        col_names = [str(name[0]).lower() for name in c.description]
    return c.fetchall(), col_names

def tokenize(text, lower=True):
    '''Simple tokenizer. Will split text on word boundaries, eliminating apostrophes and retaining alpha tokens with an exception for underscores.'''
    text = re.sub("'", "", text)
    if lower:
        tokens = re.findall('''[a-z_]+''', text.lower())
    else:
        tokens = re.findall('''[A-Za-z_]''', text)
    return tokens

def get_article(article_id):
    '''Returns tokens from a given article id. Pulls, joins, and tokenizes section text from a given article id.'''
    select = '''select section_text from articles where article_id=%d''' % article_id
    docs, _ = get_query(select)
    docs = [doc[0] for doc in docs]
    doc = ' '.join(docs)
    tokens = tokenize(doc)
    return tokens

select = '''select * from articles limit 10'''
data, cols = get_query(select)
df = pd.DataFrame(data, columns=cols)
df
tokens = get_article(0)
print(tokens[:100])
