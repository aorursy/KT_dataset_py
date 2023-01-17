import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Python defacto plotting library

%matplotlib inline 
movie_data = pd.read_csv('../input/movie_metadata.csv') # read the movie data

movie_data.drop_duplicates(inplace=True) # remove duplicate rows

movie_data.head()
from nltk import pos_tag # function that tags words by their part of speech (POS)
tagged_movie_titles = movie_data['movie_title'].str.split().map(pos_tag)

tagged_movie_titles.head()
def count_tags(title_with_tags):

    tag_count = {}

    for word, tag in title_with_tags:

        if tag in tag_count:

            tag_count[tag] += 1

        else:

            tag_count[tag] = 1

    return(tag_count)

tagged_movie_titles.map(count_tags).head()
tagged_movie_titles = pd.DataFrame(tagged_movie_titles)

tagged_movie_titles['tag_counts'] = tagged_movie_titles['movie_title'].map(count_tags)

tagged_movie_titles.head()
# list of part-of-speech tags used in the Penn Treebank Project: 

# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

tag_set = list(set([tag for tags in tagged_movie_titles['tag_counts'] for tag in tags]))

for tag in tag_set:

    tagged_movie_titles[tag] = tagged_movie_titles['tag_counts'].map(lambda x: x.get(tag, 0))

title = 'Frequency of POS Tags in Movie Titles'    

tagged_movie_titles[tag_set].sum().sort_values().plot(kind='barh', logx=True, figsize=(12,8), title=title)
vocabulary = {}

for row in tagged_movie_titles['movie_title']:

    for word, tag in row:

        if word in vocabulary:

            if tag in vocabulary[word]:

                vocabulary[word][tag] += 1

            else:

                vocabulary[word][tag] = 1

        else:

            vocabulary[word] = {tag: 1}

vocabulary_df = pd.DataFrame.from_dict(vocabulary, orient='index')

vocabulary_df.fillna(value=0, inplace=True)

tag = 'NNP' # NNP: Proper noun, singular 

vocabulary_df.sort_values(by=tag, ascending=False).head(10) # top 10 words for a given tag
size = 25

tag = 'VBG' # VBG: Verb, gerund or present participle

title = 'Top {} Most Frequent Words for {} Tag'.format(size, tag)

vocabulary_df[tag].sort_values().tail(size).plot(kind='barh', figsize=(12,6), title=title)
vocab = {}

for row in tagged_movie_titles['movie_title']:

    for word, tag in row:

        if word in vocab:

            vocab[word] += 1

        else:

            vocab[word] = 1



vocab_df = pd.DataFrame.from_dict(vocab, orient='index')

vocab_df.columns = ['count']

size = 30

title = 'Top {} Most Frequent Words in Movie Titles'.format(size)

vocab_df.sort_values(by='count').tail(size).plot(kind='barh', logx=True, figsize=(12,8), title=title)
def generate_ngrams(text, n=2):

    words = text.split()

    iterations = len(words) - n + 1

    for i in range(iterations):

       yield words[i:i + n]
n = 3 # n is the length of the ngram, 2 = bigram, 3 = trigram, etc.

ngrams = {}

for title in movie_data['movie_title']:

    for ngram in generate_ngrams(title, n):

        ngram = ' '.join(ngram)

        if ngram in ngrams:

            ngrams[ngram] += 1

        else:

            ngrams[ngram] = 1



ngrams_df = pd.DataFrame.from_dict(ngrams, orient='index')

ngrams_df.columns = ['count']

size = 25

title = 'Top {} Most Frequent {}-grams in Movie Titles'.format(size, n)

ngrams_df.sort_values(by='count').tail(size).plot(kind='barh', logx=False, figsize=(12,10), title=title)