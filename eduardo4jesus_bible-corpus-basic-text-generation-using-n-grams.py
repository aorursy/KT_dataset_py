import matplotlib.pyplot as plt

import nltk

import numpy as np

import pandas as pd

import re



from collections import Counter

from path import Path

from wordcloud import WordCloud
df_books_en = pd.read_csv('../input/key_english.csv')

df_ab_en = pd.read_csv('../input/key_abbreviations_english.csv')

df_versions = pd.read_csv('../input/bible_version_key.csv')

df_versions = df_versions.dropna(axis=1, how='all')



display(df_books_en.head())

display(df_ab_en.head())

display(df_versions)
df_bibles = list()

for index, row in df_versions.iterrows():

    df_bible = pd.read_csv(f'../input/{row.table}.csv', encoding='latin')

    df_bible['version'] = row.abbreviation

    df_bible.id = df_bible.id.apply(lambda x: int(str(row.id) + str(x)))

    df_bibles.append(df_bible)

    

df_bible_raw = pd.concat(df_bibles).set_index('id')

df_bible_raw.head()
df_bible_raw.groupby(['version']).b.nunique()
pd.set_option('max_colwidth', 800)
def get_verse(book, c, v):

    def find_name(book_name):

        selection = df_books_en.n==book_name

        if selection.sum() == 0: return 0

        return df_books_en.b[selection].iloc[0]



    def find_abv(book_abv):

        selection = df_ab_en.a==book_abv

        if selection.sum() == 0: return 0

        return df_ab_en.b[selection].iloc[0]

    

    df = df_bible_raw

    b = find_name(book) + find_abv(book)

    if b == 0: return 'Book not found'

    return df[(df.b==b) & (df.c==c) & (df.v==v)]
get_verse('Gen', 1, 1)
def get_comments(x):

    notes = re.findall('{(.*)}', x)

    return None if len(notes) == 0 else '\n'.join(notes)



comment = get_verse('Gen', 1, 1).copy()

comment.t = comment.t.apply(get_comments)

comment
df_annotations = df_bible_raw.copy()

df_annotations.t = df_annotations.t.apply(get_comments)

df_annotations = df_annotations.dropna()

df_annotations.version.value_counts().to_frame('Num. Annotation')
def del_comments(x):

    return re.sub('{(.*)}', '', x)



df_bible = df_bible_raw.copy()

df_bible.t = df_bible.t.apply(del_comments)



verse = get_verse('Gen', 1, 1).copy()

verse.t = verse.t.apply(del_comments)

verse
df_bible.t = df_bible.t.apply(lambda x: ' '.join(re.findall('\w+', x)).upper())

df_bible.head()
unigrams = [word for v in df_bible.t for word in v.split()]

unigrams = nltk.FreqDist(unigrams)

unigrams['#START'] = unigrams['#END'] = len(df_bible)

display(unigrams)
plt.figure(figsize=(20, 5))

unigrams.plot(50)

plt.savefig('unigrams.png')
verses = [['#START'] + [word for word in v.split()] + ['#END'] for v in df_bible.t]

bigrams = nltk.FreqDist([b for v in verses for b in list(nltk.bigrams(v))])

display(bigrams)
plt.figure(figsize=(20, 5))

bigrams.plot(50)

plt.savefig('bigrams.png')
verses = [['#START'] + v + ['#END'] for v in verses]

trigrams = nltk.FreqDist([b for v in verses for b in list(nltk.trigrams(v))])

display(trigrams)
plt.figure(figsize=(20, 5))

trigrams.plot(50)

plt.savefig('trigrams.png')
np.random.seed(0)
def get_random_term(freqdist, keys=None):

    pivot = np.random.rand()

    acc = 0.

    if keys is None: 

        keys = freqdist.keys()

    else:

        pivot = pivot * np.sum([freqdist.freq(k) for k in keys])

        

    for key in keys:

        acc = acc + freqdist.freq(key) 

        if pivot < acc: return key



    return None



def unigram_generator(prev=None):

    verse = prev.upper().split() if prev else list()

    while(True):

        word = get_random_term(unigrams)

        if word == '#START': continue

        elif word == '#END': break

        else: verse.append(word) 

    return ' '.join(verse)
%%time

unigram_generator()
def get_max_term(freqdist, subset=None):

    max_value = 0.

    max_key = None

    

    if subset is None: 

        subset = freqdist.keys() 

        

    for key in subset:

        value = freqdist.freq(key)

        if value > max_value:

            max_value, max_key = value, key



    return max_key



# works with both bigrams and trigrams

def select_keys(freqdist, start: tuple):

    assert(type(start) is tuple)

    return [k for k in freqdist.keys() if start == k[:len(start)]]

    

def bigram_generator(prev=None, stochastic=True, max_length=100):

    verse = prev.upper().split() if prev else list()

    prev = (verse[-1],) if prev else ('#START', )

    for i in range(max_length):

        keys = select_keys(bigrams, prev)

        func = get_random_term if stochastic else get_max_term

        curr = func(bigrams, keys)[-1]

        if curr == '#END': break

        verse.append(curr)

        prev = (curr, )

    return ' '.join(verse)
%%time

bigram_generator(stochastic=False)
%%time

bigram_generator('Jesus', stochastic=False)
%%time

bigram_generator()
%%time

bigram_generator('Jesus')
def trigram_generator(prev=None, stochastic=True, max_length=100):

    verse = prev.upper().split() if prev else list()

    prev = (verse[-2], verse[-1],) if prev else ('#START', '#START', )

    for i in range(max_length):

        keys = select_keys(trigrams, prev)

        func = get_random_term if stochastic else get_max_term

        curr = func(trigrams, keys)[-1]

        if curr == '#END': break

        verse.append(curr)

        prev = (prev[1], curr, )

    return ' '.join(verse)
%%time

trigram_generator(stochastic=False)
trigram_generator('Jesus is', stochastic=False)
%%time

trigram_generator()
%%time

trigram_generator('Jesus')