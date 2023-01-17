'''--- Libraries ---'''



# Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings, datetime, pickle, re

warnings.filterwarnings("ignore")





# Plotting

import plotly.express as px

import plotly.graph_objects as go

#from pyLDAvis import sklearn as sklda

import pyLDAvis.gensim 

import pyLDAvis.sklearn



#Gensim Library for Text Processing

import gensim.parsing.preprocessing as gsp

from gensim import utils, corpora, models



# SK Learn

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation as LDA



# Transformer

from transformers import TFAutoModelWithLMHead, AutoTokenizer, pipeline



# Spacy

import spacy

from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm',disable=['ner','textcat'])
'''--- Data ---'''



# Load

url = '../input/speeches-modi/PM_Modi_speeches.csv'

data = pd.read_csv(url, header='infer')



# Convert date column datatype to datetime

data['date'] = pd.to_datetime(data['date'], errors='ignore')



# Extract Year

data['year'] = pd.DatetimeIndex(data['date']).year



# Selecting only english speeches delivered in 2020

df = data[(data['year'] == 2020) & (data['lang'] =='en')]



# Dropping Unwanted Columns

df.drop(['lang','year', 'url'], axis=1, inplace=True)



# Total Speeches

print("Total Speeches Made in 2020: ", df.shape[0])



# Inspect

df.head()
# Visualize

fig = px.line(df, x='date', y='words', title="Speeches made by P.M Mr Modi in 2020")

fig.show()
'''Selecting Records''' 

word_count = 20000

df_20k = df[df['words'] >= word_count]

df_20k.reset_index(drop=True, inplace=True)







'''Text Cleaning Utility Function'''



processes = [

               gsp.strip_tags, 

               gsp.strip_punctuation,

               gsp.strip_multiple_whitespaces,

               gsp.strip_numeric,

               gsp.remove_stopwords, 

               gsp.strip_short #, 

               #gsp.stem_text,

               #utils.tokenize

            ]



# Utility Function

def proc_txt(txt):

    text = txt.lower()

    text = utils.to_unicode(text)

    for p in processes:

        text = p(text)

    return text



# Applying the function to text column

df_20k['cleaned_txt'] = df_20k['text'].apply(lambda x: proc_txt(x))





# Inspect

df_20k.head()
# Initialise the count vectorizer with the English stop words

count_vectorizer = CountVectorizer(stop_words='english')



# Fit and transform the processed titles

count_data = count_vectorizer.fit_transform(df_20k['cleaned_txt'])





# Parameters

num_topics = 5    # can be changed

num_words = 20    # can be changed





# Utility Function

def topics (model, vectors,num_top_wrds):

    words = count_vectorizer.get_feature_names()

    

    print(f"*** {num_topics} TOPICS DISPLAYED WITH {num_top_wrds} WORDS ***\n")

    

    for topic_idx, topic in enumerate(model.components_):

        print("Topic Index: %d" %topic_idx)

        print(" ".join([words[i]

                        for i in topic.argsort()[:-num_top_wrds - 1:-1]]), "\n")



# LDA Model

lda = LDA(n_components=num_topics, n_jobs=-1, random_state=42, verbose=0)

lda.fit(count_data)

              

# Topics Detected by LDA Model

topics(lda, count_vectorizer, num_words)
# Visualize Topics

LDAvis_data_filepath = os.path.join('./ldavis_prepared_'+str(num_topics))

LDAvis_prepared = pyLDAvis.sklearn.prepare(lda, count_data, count_vectorizer)

pyLDAvis.display(LDAvis_prepared)
'''Text Cleaning Utility Function'''



processes = [

               gsp.strip_tags, 

               gsp.strip_punctuation,

               gsp.strip_multiple_whitespaces,

               gsp.strip_numeric,

               gsp.remove_stopwords, 

               gsp.strip_short, 

               #gsp.stem_text,

               utils.tokenize

            ]



# Utility Function

def proc_txt(txt):

    text = txt.lower()

    text = utils.to_unicode(text)

    for p in processes:

        text = p(text)

    return list(text)



# Applying the function to text column

df_20k['cleaned_txt'] = df_20k['text'].apply(lambda x: proc_txt(x))
# Dictionary & Corpus

dictionary = corpora.Dictionary(df_20k['cleaned_txt'])

corpus = [dictionary.doc2bow(txt) for txt in df_20k['cleaned_txt']]



# Saving the dictionary

pickle.dump(corpus, open('corpus.pkl', 'wb'))

dictionary.save('dictionary.gensim')
# Gensim LDA Model

lda_model = models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=20)

lda_model.save("ldamodel.gensim")



# Display Topics

print(f"*** {num_topics} TOPICS DISPLAYED WITH {num_words} WORDS ***\n")

topics = lda_model.print_topics(num_words=num_words)

for topic in topics:

    print(topic,"\n")

    

# Save Dictionary

dictionary = corpora.Dictionary.load('dictionary.gensim')

corpus = pickle.load(open('corpus.pkl', 'rb'))

lda = models.ldamodel.LdaModel.load('ldamodel.gensim')



# Visualize Topics

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)

pyLDAvis.display(lda_display)
# Garbage Collection

gc.collect()
'''Select Text'''

article = df_20k['text'].iloc[1]





'''Text Cleaning Utility Function'''



processes = [

               gsp.strip_tags, 

               gsp.strip_punctuation,

               gsp.strip_multiple_whitespaces,

               gsp.strip_numeric,

               gsp.remove_stopwords, 

               gsp.strip_short

            ]



# Utility Function

def proc_txt(txt):

    text = txt.lower()

    text = utils.to_unicode(text)

    for p in processes:

        text = p(text)

    return text



# Cleaning the article

article = proc_txt(article)
# Instantiate Model

model = TFAutoModelWithLMHead.from_pretrained("bert-large-cased")

tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
# Define Input

input = tokenizer.encode(article, return_tensors="tf", max_length=256)

output = model.generate(input, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,early_stopping=True)

print("Summarized Text: \n", tokenizer.decode(output[0], skip_special_tokens=True))
# Garbage Collection

gc.collect()