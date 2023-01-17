# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd

import re

import os

from collections import Counter

import numpy as np



pd.set_option('max_columns', None)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

import plotly.offline as offline

import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

from IPython.display import IFrame

from IPython.core.display import display, HTML





from textblob import TextBlob

import scattertext as st

import pyLDAvis.gensim

from nltk.tokenize.toktok import ToktokTokenizer

from nltk.stem.wordnet import WordNetLemmatizer 

from nltk.corpus import stopwords

from nltk.corpus import wordnet

import gensim

from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

from gensim.models.wrappers import LdaMallet

from gensim.corpora import Dictionary

eng_stopwords = set(stopwords.words("english"))

import warnings

warnings.filterwarnings("ignore")

lem = WordNetLemmatizer()

tokenizer=ToktokTokenizer()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import gc

gc.collect()



from tqdm import tqdm

tqdm.pandas()



# Any results you write to the current directory are saved as output.
#Spacy

import string

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

punctuations = string.punctuation

from nltk.stem import WordNetLemmatizer 

lemmatizer = WordNetLemmatizer() 

own_stop = ['is','the','are','a','be','he','what'] 

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS





#nlp = spacy.load('en')

nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])

parser = English()



data = pd.read_csv("../input/zomato.csv",low_memory=True) 

data_full = data.copy()

import gc

gc.collect()
data['review_word_count'] = data['reviews_list'].str.split().str.len()

sns.boxplot(x=data['review_word_count'])
data = data[data['review_word_count'] < 200]
sns.boxplot(x=data['review_word_count'])
data.duplicated().sum()

data.name = data.name.apply(lambda x: x.title())

data.rate.replace(('NEW','-'),np.nan,inplace =True)

data.rate = data.rate.astype('str')

data.rate = data.rate.apply(lambda x: x.replace('/5','').strip())

data.rate = data.rate.astype('float')

data.dropna(how ='any', inplace = True)
def cleanText(inputString):

	review=re.sub(r"http\S+",'' , inputString)

	review=re.sub(r'\W',' ',review) # remove punchations 

	review=review.lower()

	review=re.sub(r'\s+[a-z]\s+',' ',review) # remove single characters which have space in starting and end of the characters

	review=re.sub(r'^[a-z]\s+',' ',review) # remnove single characters which have at starting position of the sentence 

	review=re.sub(r'^[0-9 ]+',' ',review) # remnove single characters which have at starting position of the sentence 

	review=re.sub(r'[^a-zA-Z0-9\s]',' ',review) # remove extra spaces.



	review=re.sub(r'\s+',' ',review) # remove extra spaces.

	review=[lemmatizer.lemmatize(word) for word in review.split()]

	review =' '.join(review)

	return review



def rated_Clean(inputString):

    review=re.sub(r'\brated\b','',inputString)

    return review









#Spacy Lemma # Own Stop words



def spacy_lemma_text(text):

    doc = nlp(text)

    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

    tokens = [tok for tok in tokens if tok not in spacy_stopwords and tok not in punctuations]

    tokens = ' '.join(tokens)

    return tokens
##Spacy Stemming

data['Clean_Review'] = data['reviews_list'].progress_apply(spacy_lemma_text)

data['Clean_Review'] = data['Clean_Review'].progress_apply(lambda x:cleanText(x))

data['Clean_Review'] = data['Clean_Review'].progress_apply(spacy_lemma_text)

gc.collect()
def detect_polarity(text):

    return TextBlob(text).sentiment.polarity



data['texblo_polarity'] = data.Clean_Review.apply(detect_polarity)

data['sentiment_type']=''

data.loc[data.texblo_polarity >0.25,'sentiment_type']='POSITIVE'

data.loc[data.texblo_polarity==0,'sentiment_type']='NEUTRAL'

data.loc[data.texblo_polarity<0.25,'sentiment_type']='NEGATIVE'

data['texblo_polarity'].hist(bins=50)
x=data['sentiment_type'].value_counts()

plt.figure(figsize=(9,7))

ax= sns.barplot(x.index, x.values, alpha=0.8)

plt.title("# per sentiment")

plt.ylabel('# Review Sentiment Count', fontsize=12)

plt.xlabel('Sentiment ', fontsize=12)

rects = ax.patches

labels = x.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')



plt.show()
from tqdm import tqdm

tqdm.pandas()

import scattertext as st

nlp = spacy.load('en',disable_pipes=["tagger","ner"])

data['parsed'] = data.Clean_Review.progress_apply(nlp)
corpus = st.CorpusFromParsedDocuments(data,

                             category_col='sentiment_type',

                             parsed_col='parsed').build()
html = st.produce_scattertext_explorer(corpus,

           category='POSITIVE',                            

          category_name='POSITIVE',

          not_category_name='NEGATIVE',

          width_in_pixels=600,

          minimum_term_frequency=15,

          term_significance = st.LogOddsRatioUninformativeDirichletPrior(),

          )
filename = "Postive-vs-Negative.html"

open(filename, 'wb').write(html.encode('utf-8'))

IFrame(src=filename, width = 1000, height=700)

#Using log scales

gc.collect()
html = st.produce_scattertext_explorer(corpus,

                                       category='POSITIVE',

                                       category_name='POSITIVE',

                                       not_category_name='NEGATIVE',

                                       minimum_term_frequency=15,

                                       width_in_pixels=700,

                                       transform=st.Scalers.log_scale_standardize)
filename = "Postive-vs-Negative.html"

open(filename, 'wb').write(html.encode('utf-8'))

IFrame(src=filename, width = 1900, height=700)

#to seperate sentenses into words

def preprocess(comment):

    """

    Function to build tokenized texts from input comment

    """

    return gensim.utils.simple_preprocess(comment, deacc=True, min_len=3)



all_text=data.Clean_Review.apply(lambda x: preprocess(x))

bigram = gensim.models.Phrases(all_text)

def clean(word_list):

    """

    Function to clean the pre-processed word lists 

    

    Following transformations will be done

    1) Stop words removal from the nltk stopword list

    2) Bigram collation (Finding common bigrams and grouping them together using gensim.models.phrases)

    3) Lemmatization (Converting word to its root form : babies --> baby ; children --> child)

    """

    #remove stop words

    clean_words = [w for w in word_list if not w in spacy_stopwords]

    #collect bigrams

    clean_words = bigram[clean_words]

    #Lemmatize

    clean_words=[lem.lemmatize(word, "v") for word in clean_words]

    return(clean_words)    
all_text=all_text.apply(lambda x:clean(x))

dictionary = Dictionary(all_text)

corpus = [dictionary.doc2bow(text) for text in all_text]

#create the LDA model

ldamodel = LdaModel(corpus=corpus, num_topics=5, id2word=dictionary,alpha='auto',passes=25)
pyLDAvis.enable_notebook()

#The size of the circle represents what % of the corpus it contains.

pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)

## Lets Do Some EDA !!!

data = data_full.copy()
data.rate.replace(('NEW','-'),np.nan,inplace =True)

data.rate = data.rate.astype('str')

data.rate = data.rate.apply(lambda x: x.replace('/5','').strip())

data.rate = data.rate.astype('float')



data.dropna(how ='any', inplace = True)
data.rename(columns={'approx_cost(for two people)': 'Cost_For_Two', \

                     'listed_in(city)': 'Listed_Location','listed_in(type)': 'meal_type'}, inplace=True)

labels = list(data.location.value_counts().index)

values = list(data.location.value_counts().values)



fig = {

    "data":[

        {

            "labels" : labels,

            "values" : values,

            "hoverinfo" : 'label+percent',

            "domain": {"x": [0, .9]},

            "hole" : 0.6,

            "type" : "pie",

            "rotation":120,

        },

    ],

    "layout": {

        "title" : "Zomato Bangalore",

        "annotations": [

            {

                "font": {"size":10},

                "showarrow": True,

                "text": "Locations",

                "x":0.2,

                "y":0.9,

            },

        ]

    }

}



iplot(fig)
data.Cost_For_Two = data['Cost_For_Two'].apply(lambda x: x.replace(',','').strip())

f, ax = plt.subplots(1,1, figsize = (15, 4))



ax = sns.countplot(data['rate'])

data['Cost_For_Two'] = data['Cost_For_Two'].astype(int)

data['Cost_For_Two_log'] = np.log1p(data['Cost_For_Two'])

fig, ax = plt.subplots(figsize = (16, 6))

plt.subplot(1, 2, 1)

plt.hist(data['Cost_For_Two']);

plt.title('Distribution of Cost')
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(data['Cost_For_Two'], data['rate'])

plt.title('Cost vs Rating');

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(data['Cost_For_Two_log'], data['votes'])

plt.title('Cost(log) vs Votes');

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

plt.scatter(data['rate'], data['votes'])

plt.title('Rating vs Votes');

sns.countplot(data['online_order'])
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

sns.boxplot(x='book_table', y='Cost_For_Two', data=data.loc[data['book_table'].isin(data['book_table'].value_counts().head(10).index)]);

plt.title('Book Table vs Cost');

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

sns.boxplot(x='online_order', y='Cost_For_Two', data=data.loc[data['online_order'].isin(data['online_order'].value_counts().head(10).index)]);

plt.title('Online vs Cost');

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)

sns.boxplot(x='Cost_For_Two', y='rate', data=data.loc[data['Cost_For_Two'].isin(data['Cost_For_Two'].value_counts().head(10).index)]);

plt.title('Cost vs Rating');

"""needed_col = ['rate','votes','texblo_polarity']

temp_df=data_full[needed_col]

corr=temp_df.corr()

plt.figure(figsize=(12,8))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values, annot=True)"""