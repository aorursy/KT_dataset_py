# Overall tools

import numpy as np

import pandas as pd

from scipy.spatial.distance import cdist

import sys

import os

from pathlib import Path

from pprint import pprint



# Progress bar for the loops

import time

import sys

import tqdm



# Text tools

import re, nltk, spacy, gensim

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem import PorterStemmer

from nltk.tokenize import RegexpTokenizer

nltk.download("punkt")

nltk.download("stopwords")

nltk.download('wordnet')



# Gensim

import gensim, spacy, logging, warnings

import gensim.corpora as corpora

from gensim.utils import lemmatize, simple_preprocess

from gensim.models import CoherenceModel



# NLTK Stop words

import nltk

from nltk.corpus import stopwords



#plotting tools

import pyLDAvis

import pyLDAvis.gensim

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

%matplotlib inline

from wordcloud import WordCloud, STOPWORDS



# Plotting tools

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Bokeh

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import column

from bokeh.models import RadioButtonGroup

from bokeh.models import TextInput

from bokeh.layouts import gridplot

from bokeh.models import Div

from bokeh.models import Paragraph

from bokeh.layouts import column, widgetbox
p_df = pd.read_csv('../input/nips-2015-papers')
p_df.head()
len(p_df)
# Creating a list of stopwords in english



english_stopwords = list(set(stopwords.words('english')))
# Creating a lemmatizing function



lmtzr = WordNetLemmatizer()
# Creating a stem function



porter = PorterStemmer()
# Creating a function that cleans text of special characters



def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t
# Creating a function that makes text lowercase and uses the function created above



def clean(text):

    t = text.lower()

    t = strip_characters(t)

    return t
# Tokenize into individual tokens - words mostly



def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words 

                     if len(word) > 1

                     and not word in english_stopwords

                     and not (word.isnumeric() and len(word) is not 4)

                     and (not word.isnumeric() or word.isalpha())] )

               )
# Creating a function that cleans, lemmatize and tokenize texts



def preprocess(text):

    t = clean(text)

    tokens = tokenize(t)

    l = [lmtzr.lemmatize(word) for word in tokens]

    return tokens# Creating a function that cleans, lemmatize and tokenize texts
def stemming(text):

    stem_sentence=[]

    for word in text:

        stem_sentence.append(porter.stem(word))

    return "".join(stem_sentence)
# Creating a list of stopwords in english



stop_words = list(set(stopwords.words('english')))

stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say',

                   'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather',

                   'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
# Preprocessing all the strings inside the column abstract. It will make them lowercase, remove special characters, stopwords and tokenize them.

p_df['PaperText_Processed'] = p_df['PaperText'].apply(lambda x: preprocess(x))
abstract = p_df['PaperText_Processed'].tolist()

len(abstract)
id2word = corpora.Dictionary(abstract)
corpus = [id2word.doc2bow(text) for text in abstract]
from gensim.models.ldamulticore import LdaMulticore
lda = LdaMulticore(corpus, id2word=id2word, num_topics=3)
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model = LdaMulticore(corpus, id2word=id2word, num_topics=3)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=id2word,

                                                        corpus=corpus, texts=abstract, start=1, limit=10, step=1)
# Show graph

limit=10; start=1; step=1;

x = range(start, limit, step)

plt.plot(x, coherence_values)

plt.xlabel("Num Topics")

plt.ylabel("Coherence score")

plt.legend(("coherence_values"), loc='best')

plt.show()

# Print the coherence scores

for m, cv in zip(x, coherence_values):

    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
lda_model = LdaMulticore(corpus=corpus,

                        id2word=id2word,

                        num_topics=6, 

                        random_state=42,

                        chunksize=150,

                        passes=30,

                        iterations = 500,

                        per_word_topics=True)
# Compute Perplexity (lower = better)



print('\nPerplexity: ', lda_model.log_perplexity(corpus))
# Compute Coherence Score



coherence_model_lda = CoherenceModel(model=lda_model, texts=abstract, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Visualize the topics



pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis

# Select the model and print the topics



optimal_model = model_list[3]

model_topics = optimal_model.show_topics(formatted=False)

pprint(optimal_model.print_topics(num_words=10))
# Find the dominant topic in each sentence



def format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=abstract):

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
df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=abstract)
# Format



df_dominant_topic = df_topic_sents_keywords.reset_index()

df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
# Show



df_dominant_topic.head(10)
# Group top 5 sentences under each topic

sent_topics_sorteddf_mallet = pd.DataFrame()



sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')



for i, grp in sent_topics_outdf_grpd:

    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 

                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 

                                            axis=0)



# Reset Index    

sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)



# Format

sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]



# Show

sent_topics_sorteddf_mallet
# extracting keywords per topic



keywords = sent_topics_sorteddf_mallet[['Topic_Num', 'Keywords']]

keywords.to_csv('keywords.csv')
all_keywords = keywords['Keywords'].values.tolist()

all_keywords = [i.split(',')[1] for i in all_keywords]

all_keywords_unique = keywords['Keywords'].unique().tolist()

count_keywords = len(all_keywords)

count_keywords_svg = keywords['Keywords'].nunique()
all_keywords
# Number of Documents for Each Topic

topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()



# Percentage of Documents for Each Topic

topic_contribution = round(topic_counts/topic_counts.sum(), 4)



# Topic Number and Keywords

topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]



# Concatenate Column wise

df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)



# Change Column names

df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']



# Show

df_dominant_topics
# Creating word clouds for each topic



cols = [color for name, color in mcolors.XKCD_COLORS.items()]



cloud = WordCloud(stopwords=stop_words,

                  background_color='white',

                  width=2500,

                  height=1800,

                  max_words=10,

                  colormap='tab10',

                  color_func=lambda *args, **kwargs: cols[i],

                  prefer_horizontal=1.0)



topics = lda_model.show_topics(formatted=False)



fig, axes = plt.subplots(3, 2, figsize=(10,10), sharex=True, sharey=True)



for i, ax in enumerate(axes.flatten()):

    fig.add_subplot(ax)

    topic_words = dict(topics[i][1])

    cloud.generate_from_frequencies(topic_words, max_font_size=300)

    plt.gca().imshow(cloud)

    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))

    plt.gca().axis('off')





plt.subplots_adjust(wspace=0, hspace=0)

plt.axis('off')

plt.margins(x=0, y=0)

plt.tight_layout()

plt.show()
lda = LdaMulticore(corpus, id2word=id2word, num_topics=5)
def compute_coherence_values(dictionary, corpus, texts, limit, start=1, step=1):

    coherence_values = []

    model_list = []

    for num_topics in range(start, limit, step):

        model = LdaMulticore(corpus, id2word=id2word, num_topics=5)

        model_list.append(model)

        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')

        coherence_values.append(coherencemodel.get_coherence())



    return model_list, coherence_values
model_list, coherence_values = compute_coherence_values(dictionary=id2word,

                                                        corpus=corpus, texts=abstract, start=1, limit=10, step=1)
lda_model = LdaMulticore(corpus=corpus,

                        id2word=id2word,

                        num_topics=5, 

                        random_state=42,

                        chunksize=150,

                        passes=30,

                        iterations = 500,

                        per_word_topics=True)
# Compute Perplexity (lower = better)



print('\nPerplexity: ', lda_model.log_perplexity(corpus))
# Compute Coherence Score



coherence_model_lda = CoherenceModel(model=lda_model, texts=abstract, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('\nCoherence Score: ', coherence_lda)
# Visualize the topics



pyLDAvis.enable_notebook()

vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)

vis