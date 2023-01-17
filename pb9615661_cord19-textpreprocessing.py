import pandas as pd

import string

import numpy as np

from nltk.corpus import stopwords

from collections import Counter

from tqdm.notebook import tqdm

import nltk

from nltk.stem.porter import PorterStemmer

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

import re

from glob import glob

tqdm.pandas()
def print_difference(df, col1, col2, size=200, idx=None):

    '''

    Observe the difference between two columns at the given index

    '''

    if idx is None:

        idx = np.random.randint(len(df))

    

    print('-'*100)

    print(col1)

    print('-'*100)

    print(df[col1][idx][:size])

    print('-'*100)

    print(col2)

    print('-'*100)

    print(df[col2][idx][:size])

    print('-'*100)
dfs = [] 

labels = ['non_comm_use', 'comm_use', 'pmc', 'biorxiv']

for label, data in zip(labels, glob('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/*.csv')):

    print(f'Label: {label}')

    print(f'Data: {data}', end='\n\n')

    tmp_df = pd.read_csv(data)

    tmp_df['dataset'] = label

    dfs.append(tmp_df)
df = pd.concat(dfs, ignore_index=True)

df.head()
df['no_upper_text'] = df['text'].str.lower()
print_difference(df, 'text', 'no_upper_text')
def remove_urls(text):

    regex_str = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'

    url_pattern = re.compile(regex_str)

    return url_pattern.sub(r'', text)



df['no_urls'] = df['no_upper_text'].progress_apply(lambda text: remove_urls(text))
change_idx = (df['no_upper_text'] != df['no_urls'])

change_percent = 100*np.sum(change_idx)/len(df)

print(f'{change_percent:.2f}% documents have a URL in')
max_doc_length = 500

doc_lengths = df['no_upper_text'].apply(lambda x: len(x))

small_doc_idx = (doc_lengths < max_doc_length)

small_change_idx = np.where((change_idx & small_doc_idx) == 1)[0]



idx = np.random.choice(small_change_idx)

print_difference(df, 'no_upper_text', 'no_urls', size=max_doc_length, idx=idx)
stemmer = PorterStemmer()

def stem_words(text):

    return ' '.join([stemmer.stem(word) for word in text.split()])



df['cropped_text'] = df['no_urls'].progress_apply(lambda text: stem_words(text))
# lemmatizer = WordNetLemmatizer()

# wordnet_map = {'N':wordnet.NOUN, 'V':wordnet.VERB, 'J':wordnet.ADJ, 'R':wordnet.ADV}

# def lemmatize_words(text):

    # pos_tagged_text = nltk.pos_tag(text.split())

    # return ' '.join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])



# df['cropped_text'] = df['no_urls'].progress_apply(lambda text: lemmatize_words(text))
print_difference(df, 'no_urls', 'cropped_text')
def remove_punctuation(text):

    return text.translate(str.maketrans('', '', string.punctuation))



df['no_punct_text'] = df['cropped_text'].progress_apply(lambda text: remove_punctuation(text))
print_difference(df, 'cropped_text', 'no_punct_text')
stop_words = set(stopwords.words('english'))

print(list(stop_words))
word_counts = Counter()

for text in tqdm(df['no_punct_text'].values):

    for word in text.split():

        word_counts[word] += 1
freq_to_remove = 25

freq_words = set(np.array(word_counts.most_common(freq_to_remove))[:, 0])

print(list(freq_words))
rare_words = set()

min_occurence = 2

for word, count in tqdm(word_counts.most_common()[::-1]):

    if count >= min_occurence:

        break

    rare_words.add(word)



print(len(rare_words))
bad_words = stop_words | freq_words | rare_words

print(len(bad_words))
def remove_bad_words(text):

    return ' '.join([word for word in str(text).split() if word not in bad_words])



df['no_bad_text'] = df['no_punct_text'].progress_apply(lambda text: remove_bad_words(text))
print_difference(df, 'no_punct_text', 'no_bad_text')
df['text'] = df['no_bad_text']

bad_cols = ['no_upper_text', 'no_urls', 'cropped_text', 'no_punct_text', 'no_bad_text', 'no_upper_text']

df.drop(columns=bad_cols, inplace=True)
df.head()
df.to_csv('preprocessed_CORD19.csv', index=False)