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
data = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

data.head(5)
data.describe()
z = pd.Series(data.source_x.unique())

z
import matplotlib.pyplot as plt

import scattertext as st

from scattertext import CorpusFromPandas, produce_scattertext_explorer



import spacy

nlp = spacy.load('en')

null_columns=data.columns[data.isnull().any()]

data[null_columns].isnull().sum()
data.title.size

data = data[data['title'].notna()]

data['parsed_title'] = data.title.apply(nlp)
data
print("Document Count")

print(data.groupby('source_x')['title'].count())

print("Word Count")

data.groupby('source_x').apply(lambda x: x.title.apply(lambda x: len(x.split())).sum())
corpus = st.CorpusFromParsedDocuments(data, category_col='source_x', parsed_col='parsed_title').build()
from IPython.display import IFrame

IFrame('http://stackoverflow.org', width=700, height=350)

print("iframe done")
from IPython.display import IFrame

np.seterr(divide='ignore', invalid='ignore')

html = produce_scattertext_explorer(corpus,

                                    category='CZI',

                                    category_name='CZI',

                                    not_category_name='PMC',

                                    width_in_pixels=1000,

                                    minimum_term_frequency=5,

                                    transform=st.Scalers.scale,

                                    metadata=data['has_full_text'])



open('/kaggle/working/Convention.html', 'wb').write(html.encode('utf-8'))

IFrame(src='/kaggle/working/Convention.html', width = 500, height=500)
data = data[~data['abstract'].isnull()]
from textblob import TextBlob, Word, Blobber

from textblob.classifiers import NaiveBayesClassifier

from textblob.taggers import NLTKTagger



data['abstract_polarity'] = data['abstract'].map(lambda text: TextBlob(text).sentiment.polarity)

data['abstract_review_len'] = data['abstract'].astype(str).apply(len)

data['abstract)word_count'] = data['abstract'].apply(lambda x: len(str(x).split()))
data
#import chart_studio.plotly as py

import matplotlib.pyplot as plt

#import plotly.graph_objects as go

data['abstract_review_len'].iplot(

    kind='hist',

    bins=100,

    xTitle='abstract length',

    linecolor='black',

    yTitle='count',

    title='Abstract Length Distribution')