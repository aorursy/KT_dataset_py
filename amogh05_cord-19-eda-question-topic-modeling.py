# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

from wordcloud import WordCloud, STOPWORDS

from collections import Counter

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import warnings 

import time

plt.style.use('seaborn')

import glob

import json

warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.

# load metadata

root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'

metadata_path = f'{root_path}/all_sources_metadata_2020-03-13.csv'

t1 = time.time()

df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')

t2 = time.time()

print('Elapsed time:', t2-t1)

df.info()
df.source_x.value_counts().plot(kind='bar')

plt.title("Dataset Distribution")

plt.show()



df.has_full_text.value_counts().plot(kind='bar')

plt.title("Has Full Text Distribution")

plt.show()



# plot journals

value_counts = df['journal'].value_counts()

value_counts_df = pd.DataFrame(value_counts)

value_counts_df['journal_name'] = value_counts_df.index

value_counts_df['count'] = value_counts_df['journal']

fig = px.bar(value_counts_df[0:20].sort_values('count'), 

             x="count", 

             y="journal_name",

             title='Most Common Journals in the CORD-19 Dataset',

             orientation='h')

fig.show()



# define some functions

def count_ngrams(dataframe, column, begin_ngram, end_ngram):

    # adapted from https://stackoverflow.com/questions/36572221/how-to-find-ngram-frequency-of-a-column-in-a-pandas-dataframe

    word_vectorizer = CountVectorizer(ngram_range=(begin_ngram,end_ngram), analyzer='word')

    sparse_matrix = word_vectorizer.fit_transform(df['title'].dropna())

    frequencies = sum(sparse_matrix).toarray()[0]

    most_common = pd.DataFrame(frequencies, 

                               index=word_vectorizer.get_feature_names(), 

                               columns=['frequency']).sort_values('frequency',ascending=False)

    most_common['ngram'] = most_common.index

    most_common.reset_index()

    return most_common



def word_cloud_function(df, column, number_of_words):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    word_string=str(popular_words_nonstop)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=number_of_words,

                          width=1000,height=1000,

                         ).generate(word_string)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



def word_bar_graph_function(df, column, title, nvals=50):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ x.split() for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]

    plt.barh(range(nvals), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:nvals])])

    plt.yticks([x + 0.5 for x in range(nvals)], reversed(popular_words_nonstop[0:nvals]))

    plt.title(title)

    plt.show()
# show example

df.title[0]



# show most frequent words in titles

plt.figure(figsize=(10,10))

word_bar_graph_function(df,column='title', 

                        title='Most common words in the TITLES of the papers in the CORD-19 dataset',nvals=20)
# evaluate abstract 3-grams (takes a while)

t1 = time.time()

three_gram = count_ngrams(df,'title',3,3)

t2 = time.time()

print('Elapsed time:', t2-t1)

three_gram[0:20]
# plot most frequent title 3-gram distribution

fig = px.bar(three_gram.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Most Common 3-Words in Titles of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
# title word cloud

plt.figure(figsize=(10,10))

word_cloud_function(df,column='title',number_of_words=50000)
# show most frequent words in abstracts

plt.figure(figsize=(10,10))

word_bar_graph_function(df,column='abstract',

                        title='Most common words in the ABSTRACTS of the papers in the CORD-19 dataset',

                        nvals=20)
# evaluate abstract 3-gram (takes some time)

t1 = time.time()

three_gram_abs = count_ngrams(df,'abstract',3,3)

t2 = time.time()

print('Elapsed time:', t2-t1)

three_gram_abs[0:20]
# plot most frequent abstract 3-grams distribution

fig = px.bar(three_gram_abs.sort_values('frequency',ascending=False)[0:10], 

             x="frequency", 

             y="ngram",

             title='Top Ten 3-Grams in ABSTRACTS of Papers in CORD-19 Dataset',

             orientation='h')

fig.show()
# abtract word cloud

plt.figure(figsize=(10,10))

word_cloud_function(df,column='abstract',number_of_words=50000)
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

len(all_json)
class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            self.body_text = '\n'.join(self.body_text)

            # Extend Here

            #

            #

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

first_row = FileReader(all_json[0])

print(first_row)
dict_ = {'paper_id': [], 'abstract': [], 'body_text': []}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text'])

df_covid.head()
df_covid['abstract_word_count'] = df_covid['abstract'].apply(lambda x: len(x.strip().split()))

df_covid['body_word_count'] = df_covid['body_text'].apply(lambda x: len(x.strip().split()))

df_covid.head()
df_covid.describe(include='all')
df_covid.drop_duplicates(['abstract'], inplace=True)

df_covid.describe(include='all')
col='abstract'

keep = df.dropna(subset=[col])

print(keep.shape)

docs = keep[col].tolist()
# Code adaptead from https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html



from nltk.tokenize import RegexpTokenizer

from nltk.stem.wordnet import WordNetLemmatizer

from gensim.corpora import Dictionary



tokenizer = RegexpTokenizer(r'\w+')

for idx in range(len(docs)):

    # Convert to lowercase.

    docs[idx] = docs[idx].lower()  

    # Split into words.

    docs[idx] = tokenizer.tokenize(docs[idx])  



# Remove numbers

docs = [[token for token in doc if not token.isnumeric()] for doc in docs]



# Remove one-character words

docs = [[token for token in doc if len(token) > 1] for doc in docs]



# Remove stopwords 

stop_words = stopwords.words("english")

docs = [[token for token in doc if token not in stop_words] for doc in docs]



# Lemmatize

lemmatizer = WordNetLemmatizer()

docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]



# Create a dictionary representation of the documents

dictionary = Dictionary(docs)



# Filter out words that occur less than 20 documents, or more than 50% of the documents

dictionary.filter_extremes(no_below=20, no_above=0.5)



# Create Bag-of-words representation of the documents

corpus = [dictionary.doc2bow(doc) for doc in docs]



print('Number of unique tokens: %d' % len(dictionary))

print('Number of documents: %d' % len(corpus))
from gensim.models import LdaModel, LdaMulticore



# Set training parameters.

num_topics = 10



# Make a index to word dictionary.

temp = dictionary[0]  # This is only to "load" the dictionary.

id2word = dictionary.id2token



model = LdaMulticore(

    corpus=corpus,

    id2word=id2word,

    chunksize=2000,

    eta='auto',

    iterations=10,

    num_topics=num_topics,

    passes=10,

    eval_every=None,

    workers=4

)
top_topics = model.top_topics(corpus) 

for i, (topic, sc) in enumerate(top_topics): 

    print("\nTopic {}: ".format(i) + ", ".join([w for score,w in topic]))