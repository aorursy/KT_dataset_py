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
df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df.head()
df.drop(['id', 'keyword', 'location'], inplace=True, axis=1)
df.head()
import re
def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', text)
    text_list = text.split()
    text = " ".join(text_list)
    return text.lower()

clean('ABC123#$')
df['text'] = df['text'].apply(clean)
df.head()
documents = []
for text in df['text']:
    temp = text.split()
    documents.append(temp)
from gensim.models import Word2Vec

model = Word2Vec(documents, size=150, window=10, min_count=2, workers=10, iter=10)
documents
vectors = model.wv
len(model.wv.vocab)
words = list(model.wv.vocab)
print(words)
vectors.most_similar('kill')
