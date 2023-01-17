import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from nltk.corpus import stopwords, words, qc, sentiwordnet as swn
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist, WordNetLemmatizer
from nltk import help, pos_tag, pos_tag_sents, word_tokenize

import unicodedata
from collections import defaultdict
import string
import re
import os
#print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train.set_index('qid',inplace=True,drop=True)
test.set_index('qid',inplace=True,drop=True)

train.info()
train.head(2)
# Work on subset of data for now

# Downsampled sincere q's
sincere = train[train.target==0].sample(frac=0.1)

# All insincere q's
insincere = train[train.target==1]

train = sincere.append(insincere)

train.info()

# Target distribution of subset
print(train.target.value_counts(normalize=True))
# Concatenate train and test questions
X = pd.concat([train.drop('target', axis=1), test])

# Check for distinct values
print("Total Rows:",X.index.size)
print("Distinct Rows:",X.index.nunique())

# Check for missing values
nan_cols = list(X.columns[X.isnull().any()])
print("Number of columns with NaN values:",len(nan_cols))
# Check for non-ascii characters
X['non_ascii'] = X.question_text.apply(lambda x: len(x) != len(x.encode()))

print(X['non_ascii'].value_counts(normalize=True))

X[X['non_ascii']==1].sample(2)
# Concatente all questions that contain non-ASCII chars
non_ascii_questions = X['question_text'][X['non_ascii']].str.cat()

# Dictionary to store frequency distribution
char_count = defaultdict(int)

for char in non_ascii_questions:
    try:
        char.encode('ascii')
    except UnicodeEncodeError:
        char_count[char] += 1

# Convert dictionary to DataFrame
char_count_df = pd.DataFrame(data=list(char_count.items()),
                             columns=['Char','Count'])

char_count_df['Percent of Total'] = char_count_df['Count'] / char_count_df['Count'].sum()

# Character code (for use with chr())
char_count_df['Char Code'] = char_count_df['Char'].apply(lambda x: ord(x))

char_count_df.sort_values(by='Count',
                          inplace=True,
                          ascending=False)

print('Distinct non-ASCII chars:',len(char_count.keys()))
print("Total non-ASCII chars:",char_count_df['Count'].sum())

char_count_df.head(10)
# Manually map corrected characters
corrections = {chr(8217):'\'',
               chr(8221):'"',
               chr(8220):'"',
               chr(8230):'...',
               chr(8216):'\'',
               chr(247):'/',
               chr(960):'pi',
               chr(215):'x',
               chr(8211):'-',
               chr(180):'\'',
               chr(65311):'?'}

# Map to manually corrected chars
def correct_non_ascii(question):
    result = ''
    for char in question:
        if char in corrections.keys():
            result += corrections[char]
        else:
            result += char
    return result

X.loc[X['non_ascii'],'question_text'] = X.loc[X['non_ascii'],'question_text'].apply(correct_non_ascii) 

# Correct Accents and remove other non-ASCII chars
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

X.loc[X['non_ascii'],'question_text'] = X.loc[X['non_ascii'],'question_text'].apply(remove_accented_chars)

# Check for non-ascii characters again
print("Number of non-ascii characters:",X.question_text.apply(lambda x: len(x) != len(x.encode())).sum())
X.drop('non_ascii',axis=1,inplace=True)
# Number of words
X['word_count'] = X.question_text.apply(lambda x: len(x.split()))

# Number of characters
X['char_count'] = X.question_text.apply(lambda x: len(x))

# Average word length (chars)
X['avg_word_len'] = X['char_count'] / X['word_count']

# Number of numerical characters
X['numerics_count'] = X['question_text'].apply(lambda x: len([char for char in x if char.isnumeric()]))

# Number of punctuation characters
# EOS punctuation: .?!
X['punct_count'] = X['question_text'].apply(lambda x: len([char for char in x if char in string.punctuation]))

# Number of stopwords
stop_words = stopwords.words('english')
X['stopword_count'] = X['question_text'].apply(lambda x: len([word for word in re.split(r'\W+', x) if word.lower() in stop_words]))

# Number of non-stopwords
X['non_stopword_count'] = X['word_count'] - X['stopword_count']

# Lexical diversity - word count / number of distinct words
def lexical_diversity(question):
    word_count = len(re.findall(r'\W+',question))
    vocab_size = len(set([word.lower() for word in question.split()]))
    return float(word_count/vocab_size)

X['lex_diversity'] = X['question_text'].apply(lexical_diversity)

# Number of uppercase words
def uppercase_count(question):
    uppercase_words = []
    for word in question.split():
        if word.isupper() and len(word) > 3:
            uppercase_words.append(word)
    return len(uppercase_words)

X['uppercase_count'] = X['question_text'].apply(uppercase_count)

#X['sentiment']

X.head(3)
# POS Tagging (from https://www.nltk.org/book/ch05.html)

# Tag    Meaning              English Examples
# ADJ    adjective            new, good, high, special, big, local
# ADP    adposition           on, of, at, with, by, into, under
# ADV    adverb               really, already, still, early, now
# CONJ   conjunction          and, or, but, if, while, although
# DET    determiner, article  the, a, some, most, every, no, which
# NOUN   noun                 year, home, costs, time, Africa
# NUM    numeral              twenty-four, fourth, 1991, 14:24
# PRT    particle             at, on, out, over per, that, up, with
# PRON   pronoun              he, their, her, its, my, I, us
# VERB   verb                 is, say, told, given, playing, would
# .      punctuation marks    . , ; !
# X      other                ersatz, esprit, dunno, gr8, univeristy

# Sentence tag descriptions
help.upenn_tagset()
X['pos_tag_tuples'] = pos_tag_sents(X['question_text'].apply(word_tokenize))

X['pos_tags'] = X['pos_tag_tuples'].apply(lambda x: [tup[1] for tup in x])
X.head(2)
X_graph = pd.concat(objs=[X, train[['target']]],
                    axis=1,
                    join_axes=[train.index])

# target distribution
print("Number of rows",X_graph.size)
print(X_graph.target.value_counts())
print(X_graph.target.value_counts(normalize=True))

X_graph.target.value_counts(normalize=True).plot(kind='bar',
                                                 title='Distribution of Target (%)');
# Sample insincere questions
X_graph.loc[X_graph.target==1,'question_text'].sample(5).values
# Percentiles for numerical columns
X_graph = X.merge(train[['target']],left_index=True, right_index=True)

bounds = X_graph.describe(percentiles=[.25, .5, .75, .999]).T

# Dictionary of data types and column names
dtypes_dict = X_graph.columns.to_series().groupby(X_graph.dtypes).groups
col_dtypes = {key.name:set(value) for key, value in dtypes_dict.items()}

graph_cols = (col_dtypes['int64'] - {'target'}) | col_dtypes['float64']

for col in graph_cols:
    X_graph.groupby('target')[col].plot(kind='kde',
                                        legend=True,
                                        title=col)
    plt.xlim(bounds.loc[col]['min'],bounds.loc[col]['99.9%']) # exclude outliers from density plot
    plt.show()
def preprocessing(question):
    # Convert to lower case
    cleaned = " ".join(word.lower() for word in re.split(r'\W+',question))

    # Remove punctuation
    cleaned = re.sub(r'[^\w\s]',' ',cleaned) # do we want to keep website url's?

    # Keep only alphabetical characters
    cleaned = " ".join(word for word in cleaned.split() if word.isalpha())
    
    # Remove stop words
    cleaned = " ".join(word for word in cleaned.split() if word not in stop_words)

    return cleaned

X['clean_text'] = X['question_text'].apply(preprocessing)
# # Stemming
# ps = PorterStemmer()

# def stemmer(question):
#     text = ' '.join([ps.stem(word) for word in question.split()])
#     return text

# Lemmatization
wnl = WordNetLemmatizer()

def lemmatizer(question):
    text = ' '.join([wnl.lemmatize(word) for word in question.split()])
    return text

X['clean_text'] = X['clean_text'].apply(lemmatizer)
# Remove frequent words and rare words

def cound_words(df,col):
    # Return dataframe containing frequency count of each word
    word_bag = df[col].str.cat(sep=' ')

    fdist = FreqDist([word for word in word_bag.split()])

    print('Number of distinct words:',len(fdist.keys()))

    word_count_df = pd.DataFrame(data=list(fdist.items()),
                                 columns=['Word','Count'])
    
    word_count_df.set_index(keys='Word',
                            drop=True,
                            inplace=True)

    word_count_df.sort_values(by='Count',
                              inplace=True,
                              ascending=False)
    
    return word_count_df

word_count = cound_words(X,'clean_text')

word_count.head(20)
# Remove frequent and rare words
MAX_WORD_COUNT = 8000
MIN_WORD_COUNT = 10

def keep_word(word):
    wc = word_count['Count'].loc[word]
    if MIN_WORD_COUNT < wc < MAX_WORD_COUNT:
        return True
    else:
        return False

def remove_words(question):
    text = ' '.join([word for word in question.split() if keep_word(word)])
    return text

X['clean_text'] = X['clean_text'].apply(remove_words)

X.head()