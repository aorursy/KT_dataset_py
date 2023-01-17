# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import spacy
import re
from time import time 
df = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
df.head()
df = df[['Job Title', 'Job Description']]
df.head()
df.isnull().sum()
df = df.reset_index(drop=True)
df.shape
nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)
brief_cleaning = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in df['Job Description'])
print(df['Job Description'])
t = time()

txt = [cleaning(doc) for doc in nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1)]

print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
df_clean = pd.DataFrame({'clean': txt})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.shape
df_clean.head()
import gensim 
from gensim.models import Word2Vec
sent = [row.split() for row in df_clean['clean']]
print(sent[:10])
t = time()

model = Word2Vec(sent, min_count=1,size= 50,workers=3, window =3, sg = 1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
model.wv.most_similar(positive=["data"])
model.wv.index2entity[:50] #retrieving the 50 most common words in the corpus
model.wv.similarity("sql", 'python') #how similar are thise languages?
model.wv.doesnt_match(['report', 'analysis', 'salary']) #which word doesn't fit?
