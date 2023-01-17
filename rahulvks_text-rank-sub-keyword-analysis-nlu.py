!pip install rake_nltk

!pip install pytextrank
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import re

import os

import spacy



#https://pypi.org/project/rake-nltk/

import rake_nltk

from rake_nltk import Metric, Rake

r = Rake()



#https://github.com/vi3k6i5/flashtext

from flashtext import KeywordProcessor



#TextRank

#https://towardsdatascience.com/textrank-for-keyword-extraction-by-python-c0bae21bcec0

import pytextrank



#Spacy

import spacy

nlp = spacy.load('en')

import spacy

nlp = spacy.load('en_core_web_sm')





#Bar

from tqdm import tqdm, tqdm_pandas

tqdm(tqdm())

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

print("Data shape = ",data_train.shape)

data_train.head(2)
data_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

print("Data shape = ",data_train.shape)

data_train.head(2)
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
data_train['text_clean'] = data_train['text'].apply(remove_URL)

data_train['text_clean'] = data_train['text'].apply(spacy_lemma_text)

print("Train Cleaning - Done")

data_test['text_clean'] = data_test['text'].apply(remove_URL)

data_test['text_clean'] = data_test['text'].apply(spacy_lemma_text)

print("Test Cleaning - Done")
Rake_keywords = []

r = Rake()

r = Rake(min_length=2, max_length=4)



for text in data_train['text_clean']:

      r.extract_keywords_from_text(text)

      r.get_ranked_phrases()

      Rake_keywords.append(r.get_ranked_phrases())  



data_train['Rake_Keyword'] = Rake_keywords        
Rake_keywords = []

r = Rake()

r = Rake(min_length=2, max_length=4)



for text in data_test['text_clean']:

      r.extract_keywords_from_text(text)

      r.get_ranked_phrases()

      Rake_keywords.append(r.get_ranked_phrases())  



data_test['Rake_Keyword'] = Rake_keywords     
from flashtext import KeywordProcessor

keyword_processor = KeywordProcessor(case_sensitive=False)
keyword_dict = {

    "excaltor": ["exclators", "excaltors"]}

keyword_processor.add_keywords_from_dict(keyword_dict)



## You can add the important key word i.e prouduct name , features , payments
Flash_keywords = []

for i in data_train['text_clean']:

    keyword_processor.extract_keywords(i)

    Flash_keywords.append(keyword_processor.extract_keywords(i))

    

data_train['Flash_Keyword'] = Flash_keywords    
Flash_keywords = []

for i in data_test['text_clean']:

    keyword_processor.extract_keywords(i)

    Flash_keywords.append(keyword_processor.extract_keywords(i))

    

data_test['Flash_Keyword'] = Flash_keywords        
aspect_terms = []

for review in nlp.pipe(data_train.text_clean):

    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']

    aspect_terms.append(' '.join(chunks))

    

data_train['Aspect_Terms'] = aspect_terms    
aspect_terms = []

for review in nlp.pipe(data_test.text_clean):

    chunks = [(chunk.root.text) for chunk in review.noun_chunks if chunk.root.pos_ == 'NOUN']

    aspect_terms.append(' '.join(chunks))

    

data_test['Aspect_Terms'] = aspect_terms    
sentiment_terms = []



for review in nlp.pipe(data_train['text_clean']):

        if review.is_parsed:

            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))

        else:

            sentiment_terms.append('')  

            

data_train['Sentiment_terms'] = sentiment_terms            
sentiment_terms = []



for review in nlp.pipe(data_test['text_clean']):

        if review.is_parsed:

            sentiment_terms.append(' '.join([token.lemma_ for token in review if (not token.is_stop and not token.is_punct and (token.pos_ == "ADJ" or token.pos_ == "VERB"))]))

        else:

            sentiment_terms.append('')  

            

data_test['Sentiment_terms'] = sentiment_terms            
import spacy

import pytextrank

nlp = spacy.load('en_core_web_sm')

tr = pytextrank.TextRank()

nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)
pytext_key = []



for text in data_train['text_clean']:

    text = nlp(text)

    t = text._.phrases

    pytext_key.append(t)

    

data_train['Pytextrank_keyword'] = pytext_key    
pytext_key = []



for text in data_test['text_clean']:

    text = nlp(text)

    t = text._.phrases

    pytext_key.append(t)

    

data_test['Pytextrank_keyword'] = pytext_key    
data_train.head()
data_test.head()
from IPython.core.display import display, HTML

import plotly.graph_objects as go

df = data_train.copy()

df['length'] = df['text_clean'].apply(len)
data = [

    go.Box(

        y=df[df['target']==1]['length'],

        name='Real'

    ),

    go.Box(

        y=df[df['target']==0]['length'],

        name='Not'

    ),



]

layout = go.Layout(

    title = 'Target Class Vs Comment Lenght (After Cleaning)'

)

fig = go.Figure(data=data, layout=layout)

fig.show()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.decomposition import TruncatedSVD



import seaborn as sns



def get_tf_idf(df, text_field ='text', sent_field="airline_sentiment"):

    # creating bag of words freq counts post tokenizing

    count_vect = CountVectorizer()

    X_bog = count_vect.fit_transform(df[text_field])



    # creating the tf-idf vectors. 

    tf_transformer = TfidfTransformer(norm='l2')

    X = tf_transformer.fit_transform(X_bog)

    y = df[sent_field]

    return X, y
X, y = get_tf_idf(df, "text_clean", "target")

pca = TruncatedSVD(n_components=2, n_iter=7, random_state=4)

pca.fit_transform(X.T)

ax1 = sns.scatterplot(x=pca.components_[0], y=pca.components_[1], hue=y)

plt.title("Text Vs Target")
X, y = get_tf_idf(df, "Aspect_Terms", "target")

pca = TruncatedSVD(n_components=2, n_iter=7, random_state=4)

pca.fit_transform(X.T)

ax2 = sns.scatterplot(x=pca.components_[0], y=pca.components_[1], hue=y)

plt.title("Aspect Terms Vs Target")
X, y = get_tf_idf(df, "Sentiment_terms", "target")

pca = TruncatedSVD(n_components=2, n_iter=7, random_state=4)

pca.fit_transform(X.T)

ax3 = sns.scatterplot(x=pca.components_[0], y=pca.components_[1], hue=y)

plt.title("Sentiment Terms Vs Target")
print ("Working on the Aspect Based Extraction....:) ")