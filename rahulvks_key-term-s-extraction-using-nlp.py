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
df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv").astype(str)

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv").astype(str)

submission = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv").astype(str)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import re

import os

import spacy



#https://pypi.org/project/rake-nltk/

!pip install rake_nltk



#import rake_nltk

from rake_nltk import Metric, Rake

r = Rake()



#https://github.com/vi3k6i5/flashtext

from flashtext import KeywordProcessor



#TextRank

#https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0

!pip install pytextrank

import pytextrank



#Spacy

import spacy

nlp = spacy.load('en_core_web_sm')





#Bar

from tqdm import tqdm, tqdm_pandas

tqdm(tqdm())

import matplotlib.pyplot as plt

### Own Stop words

own_stop_word = ['i','we','are','and']

### Spacy Lemma 

def spacy_lemma_text(text):

    doc = nlp(text)

    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

    tokens = [tok for tok in tokens if tok not in own_stop_word ]

    tokens = ' '.join(tokens)

    return tokens



### Remove URL

def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)
df['text_clean'] = df['text'].apply(remove_URL)

df['text_clean'] = df['text_clean'].apply(spacy_lemma_text)

print("Train Cleaning - Done")

test['text_clean'] = test['text'].apply(remove_URL)

test['text_clean'] = test['text_clean'].apply(spacy_lemma_text)

print("Test Cleaning - Done")

test['sentiment'].value_counts()
from IPython.core.display import display, HTML

import plotly.graph_objects as go

test ['length'] = test['text'].apply(len)



data = [

    go.Box(

        y=test[test['sentiment']=='positive']['length'],

        name='positive'

    ),

    go.Box(

        y=test[test['sentiment']=='negative']['length'],

        name='negative'

    ),

    go.Box(

        y=test[test['sentiment']=='neutral']['length'],

        name='neutral'

    ),



]

layout = go.Layout(

    title = 'Sentiment Vs Comment Lenght (Before pre-processing)'

)

fig1 = go.Figure(data=data, layout=layout)





test ['length'] = test['text_clean'].apply(len)



data = [

    go.Box(

        y=test[test['sentiment']=='positive']['length'],

        name='positive'

    ),

    go.Box(

        y=test[test['sentiment']=='negative']['length'],

        name='negative'

    ),

    go.Box(

        y=test[test['sentiment']=='neutral']['length'],

        name='neutral'

    ),



]

layout = go.Layout(

    title = 'Sentiment Vs Comment Lenght (After pre-processing)'

)

fig2 = go.Figure(data=data, layout=layout)

fig1.show()
fig2.show()
aspect_terms = []

for review in nlp.pipe(test.text_clean):

    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']

    aspect_terms.append(' '.join(chunks))

    

test['Aspect_Terms'] = aspect_terms    
test.sample(5)
sentiment_terms = []



for review in nlp.pipe(test['text_clean']):

        if review.is_parsed:

            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))

        else:

            sentiment_terms.append('')  

            

test['Sentiment_terms'] = sentiment_terms  
test.sample(5)
import spacy

import pytextrank

nlp = spacy.load('en_core_web_sm')

tr = pytextrank.TextRank()

nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)
pytext_key = []



for text in test['text_clean']:

    text = nlp(text)

    t = text._.phrases

    pytext_key.append(t)

    

test['Pytextrank_keyword'] = pytext_key  
test['Pytextrank_keyword'] = test['Pytextrank_keyword'].agg(lambda x: ','.join(map(str, x)))

test.sample(5)
submission['selected_text'] =  test['Aspect_Terms'] + ' ' + test['Sentiment_terms']  + ' ' +test['Pytextrank_keyword']
submission.head()
submission.shape