import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



plt.style.use('seaborn-whitegrid')



characters = pd.read_csv('../input/simpsons_characters.csv', index_col='id').sort_index()

lines = pd.read_csv('../input/simpsons_script_lines.csv', index_col='id', error_bad_lines=False).sort_index()
characters.head()
#characters[characters['normalized_name'].str.contains('homer')][:30]

characters.loc[[125, 625, 1011, 1085, 2408], ['name']]
lines.drop(['number', 'timestamp_in_ms', 'raw_text', 'normalized_text'], axis=1).head()
lines.drop(['number', 'timestamp_in_ms', 'raw_text'], axis=1).groupby('raw_character_text').count().sort_values(by='normalized_text', ascending=False)[:10]
lines.loc[(lines.raw_character_text == 'Homer Simpson') & (lines.word_count.isnull()), ['raw_text', 'raw_character_text', 'spoken_words']][:5]
fig, ax = plt.subplots(figsize=(12, 8))

lines.groupby('raw_character_text')['raw_text'].count().sort_values(ascending=False)[:40].plot(kind='bar')

ax.set_title('Number of lines for top 40 characters')

ax.set_xlabel('');
fig, ax = plt.subplots(figsize=(12, 8))

lines.groupby('raw_character_text')['word_count'].sum().sort_values(ascending=False)[:40].plot(kind='bar')

ax.set_title('Number of words for top 40 characters')

ax.set_xlabel('');
from nltk.corpus import cmudict

prondict = cmudict.dict()



def numsyllables(word):

    try:

        pron = prondict[word][0]

        return len([s for s in pron if any([char.isdigit() for char in s])])

    except KeyError:

        return 1

    

def total_sylls(x):

    return sum([numsyllables(word) for word in x['normalized_text'].split(' ')])



lines['syllable count'] = lines.dropna(subset=['normalized_text']).apply(lambda x: total_sylls(x), axis=1)

lines['sentence count'] = lines.dropna(subset=['normalized_text'])['spoken_words'].str.count('\.')



counts = lines.groupby('raw_character_text')[['word_count', 'sentence count', 'syllable count']].sum()

counts = counts.sort_values(by='word_count', ascending=False)



counts['Flesch readability'] = 206.835 - 1.015*counts['word_count']/counts['sentence count'] - 84.6*counts['syllable count']/counts['word_count']

counts['Flesch-Kincaid grade'] = 0.39*counts['word_count']/counts['sentence count'] + 11.8*counts['syllable count']/counts['word_count'] - 15.59



fig, ax = plt.subplots(1, 2, figsize=(16, 6))

counts['Flesch readability'][:30].sort_values().plot(kind='bar', ax=ax[0])

ax[0].set_title('Flesch readability')

ax[0].set_xlabel('')

counts['Flesch-Kincaid grade'][:30].sort_values().plot(kind='bar', ax=ax[1])

ax[1].set_title('Flesch-Kincaid grade level')

ax[1].set_xlabel('');
from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer



all_speech = lines.dropna(subset=['normalized_text']).groupby('raw_character_text')['spoken_words'].agg(' '.join)



stop_words = set(stopwords.words('english'))



def proc(x):

    tokenizer = RegexpTokenizer(r'\w+')

    #print(x)

    #print(sent_tokenize(x))

    #tokens = word_tokenize(sent_tokenize(x))

    return [word.lower() for word in tokenizer.tokenize(x) if word not in stop_words]



all_speech = pd.DataFrame(all_speech.apply(lambda x: proc(x)))



all_speech['total words'] = all_speech['spoken_words'].transform(len)

all_speech['vocab size'] = all_speech['spoken_words'].transform(lambda x: len(set(x)))

all_speech['vocab:total ratio'] = all_speech['vocab size'] / all_speech['total words']



all_speech.sort_values(by='total words', ascending=False, inplace=True)



all_speech.head(10)
all_speech.sort_values(by='vocab size', ascending=False).head(10)
fig, ax = plt.subplots()

ax.scatter(all_speech['total words'], all_speech['vocab size'])

ax.set_xlabel('Total Words')

ax.set_ylabel('Vocab Size');
fig, ax = plt.subplots()

ax.scatter(all_speech.loc[all_speech['total words']>500, 'total words'],

            all_speech.loc[all_speech['total words']>500, 'vocab:total ratio'])

ax.set_xlabel('Total Words')

ax.set_ylabel('Vocab:Total Ratio');
all_speech['new vocab curve'] = all_speech.ix[:30, 'spoken_words'].apply(lambda x: [len(set(x[:i])) for i in range(min(len(x), 20000))])



colors = plt.cm.nipy_spectral(np.linspace(0, 1, 10))



fig, ax = plt.subplots(figsize=(12, 12))

ax.set_xlabel('Total Words')

ax.set_ylabel('Unique Words')

for i in range(30):

    plt.plot(all_speech.ix[i, 'new vocab curve'], color=colors[i%10], label=str(all_speech.index[i]));

legend = ax.legend(fontsize='x-large', frameon=True, bbox_to_anchor=(1, 1));

legend.get_frame().set_facecolor('lightgrey')
Homer_words = []

for j in range(10):

    Homer_words.append([len(set(all_speech.loc['Homer Simpson', 'spoken_words'][17500*j:17500*j+i])) for i in range(17500)])



fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xlabel('Total Words')

ax.set_ylabel('Unique Words')

for i in range(len(Homer_words)):

    plt.plot(Homer_words[i], color=colors[i%10], label='Homer group '+str(i+1));

legend = ax.legend(fontsize='x-large', frameon=True, bbox_to_anchor=(1, 1));

legend.get_frame().set_facecolor('lightgrey')
Marge_words = []

for j in range(5):

    Marge_words.append([len(set(all_speech.loc['Marge Simpson', 'spoken_words'][15000*j:15000*j+i])) for i in range(15000)])



colors = plt.cm.nipy_spectral(np.linspace(0, 1, 5))



fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlabel('Total Words')

ax.set_ylabel('Unique Words')

for i in range(len(Marge_words)):

    plt.plot(Marge_words[i], color=colors[i%10], label='Marge group '+str(i+1));

legend = ax.legend(fontsize='x-large', frameon=True, bbox_to_anchor=(1, 1));

legend.get_frame().set_facecolor('lightgrey')
Bart_words = []

for j in range(7):

    Bart_words.append([len(set(all_speech.loc['Bart Simpson', 'spoken_words'][10000*j:10000*j+i])) for i in range(10000)])

    

Lisa_words = []

for j in range(6):

    Lisa_words.append([len(set(all_speech.loc['Lisa Simpson', 'spoken_words'][10000*j:10000*j+i])) for i in range(10000)])



colors = plt.cm.nipy_spectral(np.linspace(0, 1, 7))



fig, ax = plt.subplots(2, 1, figsize=(8, 12))

ax[0].set_xlabel('Total Words')

ax[0].set_ylabel('Unique Words')

for i in range(len(Bart_words)):

    ax[0].plot(Bart_words[i], color=colors[i%10], label='Bart group '+str(i+1));

legend = ax[0].legend(fontsize='x-large', frameon=True, bbox_to_anchor=(1, 1));

legend.get_frame().set_facecolor('lightgrey')

ax[1].set_xlabel('Total Words')

ax[1].set_ylabel('Unique Words')

for i in range(len(Lisa_words)):

    ax[1].plot(Lisa_words[i], color=colors[i%10], label='Lisa group '+str(i+1));

legend = ax[1].legend(fontsize='x-large', frameon=True, bbox_to_anchor=(1, 1));

legend.get_frame().set_facecolor('lightgrey')