import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('/kaggle/output'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

import spacy
from nltk.corpus import stopwords

# for plotting
import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
# # import nltk
# # nltk.download('stopwords')
# #python -m spacy download en
!wget http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
!unzip mallet-2.0.8.zip
# os.listdir('/kaggle/working')
IN_PATH = '/kaggle/input/enron-dataset-cleaned'
FILE_NAME = 'emails.csv'
OUT_PATH = '/kaggle/working'

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

mallet_path = os.path.join(OUT_PATH, './mallet-2.0.8/bin/mallet')


emails = pd.read_csv(os.path.join(IN_PATH, FILE_NAME))
# emails_subset = emails[:10000]
emails_subset = emails.sample(frac=0.02, random_state=1)
def parse_raw_message(raw_message):
    '''
        Funtion for cleanning each email..
    '''
    lines = raw_message.split('\n')
    email = {}
    message = ''
    keys_to_extract = ['from', 'to']
    for line in lines:
        if ':' not in line:
            message += line.strip()
            email['body'] = message
        else:
            pairs = line.split(':')
            key = pairs[0].lower()
            val = pairs[1].strip()
            if key in keys_to_extract:
                email[key] = val
    return email

def map_to_list(emails, key):
    '''
        Helper Function for parse_into_emails to wrap things up!
    '''
    results = []
    for email in emails:
        if key not in email:
            results.append('')
        else:
            results.append(email[key])
    return results

def parse_into_emails(messages):
    '''
        Function for cleaning all emails and returning them as a dictionary
    '''
    emails = [parse_raw_message(message) for message in messages]
    return {
        'body': map_to_list(emails, 'body'),
        'to': map_to_list(emails, 'to'),
        'from_': map_to_list(emails, 'from')
    }


def sent_to_words(sentences):
    '''
        # tokenize - break down each sentence into a list of words
    '''
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        

# remove stop_words, make bigrams and lemmatize
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out
## Before preprocessing
print(emails_subset.shape)
emails_subset.head()
## After preprocessing
email_df = pd.DataFrame(parse_into_emails(emails_subset.message))
print(email_df.shape)
email_df.head()
print(email_df.iloc[3]['body'])
# Convert the body of the emails to a list
data = email_df.body.values.tolist()

# Convert the list of sentence into list of words <> Tokenizing
data_words = list(sent_to_words(data))

print(data_words[3])
## Creating models for <> bigram and trigram 
bigram = Phrases(data_words, min_count=5, threshold=100)
trigram = Phrases(bigram[data_words], threshold=100)

## Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

## Trigram example
print(trigram_mod[bigram_mod[data_words[3]]])
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)


# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[3])
'''
Each time we now see a token in a text, information on its frequency is paired with it.
A word/token like contract could then be represented as (6, 3) — >(token_id, token_count).\
'''

## Creating Dictionary
id2word = corpora.Dictionary(data_lemmatized)

## Create Corpus
texts = data_lemmatized

## Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(
    corpus=corpus,    # Stream of document vectors or sparse matrix of shape (num_terms, num_documents)
    id2word=id2word,  # It is used to determine the vocabulary size, as well as for debugging and topic printing.
    num_topics=6,    # The number of requested latent topics to be extracted from the training corpus.
    random_state=100, # Useful for reproducibility.
    update_every=1,   # Set to 0 for batch learning, > 1 for online iterative learning.
    chunksize=100,    # Number of documents to be used in each training chunk.
    passes=10,        # Number of passes through the corpus during training.
    alpha='auto',     # auto: Learns an asymmetric prior from the corpus
    per_word_topics=True 
    # If True, the model also computes a list of topics, sorted in descending order of most likely topics for each word,
    # along with their phi values multiplied by the feature-length (i.e. word count)
)
print(lda_model.print_topics())# The weights reflect how important a keyword is to that topic.

doc_lda = lda_model[corpus]

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)
# Visualize the topics
pyLDAvis.enable_notebook(sort=True)
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

pyLDAvis.display(vis)
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=6, id2word=id2word)
# Show Topics
print(ldamallet.show_topics(formatted=False))
# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)
##  Converting lda mallet to lda model for visualizing with pyLDAvis
model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)

# Visualize the topics with mallet model
pyLDAvis.enable_notebook(sort=True)
vis = pyLDAvis.gensim.prepare(model, corpus, id2word)
pyLDAvis.display(vis)
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#     Compute c_v coherence for various number of topics

#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics

#     Returns:
#     -------
#     model_list : List of LDA topic models
#     coherence_values : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     coherence_values = []
#     model_list = []
#     for num_topics in tqdm(range(start, limit, step)):
#         model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())

#     return model_list, coherence_values
# # run
# model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_lemmatized, start=2, limit=40, step=6)
# # Show graph
# limit=40; start=2; step=6;
# x = range(start, limit, step)
# plt.plot(x, coherence_values)
# plt.xlabel("Num Topics")
# plt.ylabel("Coherence score")
# plt.legend(("coherence_values"), loc='best')
# plt.show()
# # Print the coherence scores
# for m, cv in zip(x, coherence_values):
#     print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
# Select the model and print the topics
# optimal_model = model_list[4]
optimal_model = ldamallet
model_topics = optimal_model.show_topics(formatted=False)
print(optimal_model.print_topics(num_words=8))
def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)
df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=data)
# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.head(10)
df_dominant_topic.shape
df_dominant_topic[:300].to_csv('final.csv')
