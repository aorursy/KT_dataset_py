import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import os

from nltk.corpus import stopwords

import string

import re

from sklearn import svm

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk import word_tokenize, pos_tag, ne_chunk, tree2conlltags
# Load the data

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df_sample = pd.read_csv("../input/sample_submission.csv")
sia = SentimentIntensityAnalyzer()

def sentiment_nltk(text):

    res = sia.polarity_scores(text)

    return res['compound']
def chars_between_commas(text):

    return np.mean([len(chunk) for chunk in text.split(",")])
def count_unknown_symbols(text):

    symbols_known = string.ascii_letters + string.digits + string.punctuation

    return sum([not x in symbols_known for x in text])
def get_persons(text):

    # Some names have family and given names, but both belong to the same person

    # Bind them!

    def bind_names(tagged_words):

        names = list()

        name = list()

        # Bind several consequtive words with 'PERSON' tag

        for i, w in enumerate(tagged_words):

            if i == 0:

                continue

            if "PERSON" in w[2]:

                name.append(w[0])

            else:

                if len(name) != 0:

                    names.append(" ".join(name))

                name = list()

        return names

        

    res_ne_tree = ne_chunk(pos_tag(word_tokenize(text)))

    res_ne = tree2conlltags(res_ne_tree)

    res_ne_list = [list(x) for x in res_ne]

    return bind_names(res_ne_list)
class WordCloudIntersection():

    

    def __init__(self, stopwords=list(), punctuation=list(), stemmer=None, ngram=1):

        self.stopwords = stopwords

        self.punctuation = punctuation

        self.remove = self.stopwords + self.punctuation

        self.clouds = dict()

        self.texts = dict()

        self.stemmer = stemmer

        self.ngram = ngram

    

    def find_ngrams(self, input_list, n):

        return [" ".join(list(i)) for i in zip(*[input_list[i:] for i in range(n)])]

    

    # It would be much  more correct to call this function 'get_tokens'

    # it extracts not only words, but n-grams as well

    def get_words(self, text):

        words = nltk.tokenize.word_tokenize(text)

        words = [w for w in words if not w in self.remove]

        if not self.stemmer is None:

            words = [self.stemmer.stem(w) for w in words]

        

        if self.ngram > 1:

            words = self.find_ngrams(words, self.ngram)

        return words

    

    # Jaccard distance again

    def relative_intersection(self, x, y):

        try:

            return len(x & y)/len(x | y)

        except:

            return 0.0

    

    def fit(self, x, categories, data_train, data_test=None):

        cat_names = np.unique(data_train[categories])

        

        text_train = " ".join(list(data_train[x]))

        text_test = ""

        if not data_test is None:

            text_test = " ".join(list(data_test[x]))

        

        # Tokens presenting in both train and test data

        words_unique = self.get_words((text_train + text_test).lower())

        

        for cat in cat_names:

            self.texts[cat] = (" ".join(list(data_train[x][data_train[categories] == cat]))).lower()

            words = self.get_words(self.texts[cat])

            self.clouds[cat] = pd.value_counts(words)

        

        # use only tokens presented in both train and test data, 

        # feature will force your model to overfit to the train data otherwise    

        for cat in cat_names:

            self.clouds[cat] = self.clouds[cat][list(set(self.clouds[cat].index) & set(words_unique))]

        

        # Keep only author-specific tokens

        for cat in cat_names:

            key_leftover = list(set(cat_names) - set([cat]))

            bigrams_other = set(self.clouds[key_leftover[0]].index) | set(self.clouds[key_leftover[1]].index)

            self.clouds[cat] = self.clouds[cat][list(set(self.clouds[cat].index) - bigrams_other)]

        

    def transform(self, x, data):

        intersection = dict()

        prefix = '_intersect_'

        if self.ngram > 1:

            prefix = '%s-gram%s' % (self.ngram, prefix)

        else:

            prefix = 'word' + prefix

        for key in self.clouds.keys():

            category_words_set = set(self.clouds[key].index)

            intersection[prefix+key] = list()

            for text in data[x]:

                unique_words = set(self.get_words(text.lower()))

                fraction = self.relative_intersection(unique_words, category_words_set)

                intersection[prefix+key].append(fraction)

        return pd.DataFrame(intersection)


# Split text into words

def get_words(text):

    words = nltk.tokenize.word_tokenize(text)

    return [word for word in words if not word in string.punctuation]



# string.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

def count_punctuation(text):

    return sum([x in string.punctuation for x in text])



def count_capitalized_words(text):

    return sum([word.istitle() for word in get_words(text)])



def count_uppercase_words(text):

    return sum([word.isupper() for word in get_words(text)])

    

def count_tokens(text, tokens):

    return sum([w in tokens for w in get_words(text)])



def first_word_len(text):

    return len(get_words(text)[0])



def last_word_len(text):

    return len(get_words(text)[-1])



def symbol_id(x):

    symbols = [x for x in string.ascii_letters + string.digits + string.punctuation]

    return np.where(np.array(symbols) == x)[0][0]



# It is not a feature! It is just Jaccard distance

def relative_len(set_x, set_y):

    return len(set_x & set_y)/len(set_x | set_y)
stopwords = nltk.corpus.stopwords.words('english')

bigci = WordCloudIntersection(stopwords=stopwords, 

                            punctuation=list(string.punctuation),

                            stemmer=nltk.stem.SnowballStemmer('english'), ngram=2)

bigci.fit(x='text', categories='author', data_train=df_train, data_test=df_test)
df_train_intersections = bigci.transform(x='text', data=df_train)

df_test_intersections = bigci.transform(x='text', data=df_test)
df_train = pd.concat([df_train, df_train_intersections], axis=1)

df_test = pd.concat([df_test, df_test_intersections], axis=1)
_, axes = plt.subplots(1, 3, figsize=(16,6))

sns.violinplot(x='author', y='2-gram_intersect_EAP', data=df_train, ax=axes[0])

sns.violinplot(x='author', y='2-gram_intersect_HPL', data=df_train, ax=axes[1])

sns.violinplot(x='author', y='2-gram_intersect_MWS', data=df_train, ax=axes[2])

plt.show()
text_EAP = " ".join(list(df_train['text'][df_train['author'] == "EAP"]))

text_HPL = " ".join(list(df_train['text'][df_train['author'] == "HPL"]))

text_MWS = " ".join(list(df_train['text'][df_train['author'] == "MWS"]))

persons_EAP = set(get_persons(text_EAP))

persons_HPL = set(get_persons(text_HPL))

persons_MWS = set(get_persons(text_MWS))

# Keep only names related to the authors without any intersections with others

persons_EAP = persons_EAP - persons_HPL - persons_MWS

persons_HPL = persons_HPL - persons_EAP - persons_MWS

persons_MWS = persons_MWS - persons_EAP - persons_HPL
for df, name in zip([df_train, df_test], ["train", "test"]):

    df['persons_EAP_frac'] = df['text'].apply(lambda x: relative_len(persons_EAP, set(get_persons(x))))

    df['persons_HPL_frac'] = df['text'].apply(lambda x: relative_len(persons_HPL, set(get_persons(x))))

    df['persons_MWS_frac'] = df['text'].apply(lambda x: relative_len(persons_MWS, set(get_persons(x))))
_, axes = plt.subplots(1, 3, figsize=(16,6))

sns.violinplot(x='author', y='persons_EAP_frac', data=df_train, ax=axes[0])

sns.violinplot(x='author', y='persons_HPL_frac', data=df_train, ax=axes[1])

sns.violinplot(x='author', y='persons_MWS_frac', data=df_train, ax=axes[2])

plt.show()
for df, name in zip([df_train, df_test], ["train", "test"]):

    print("Generating features for %s..." % name)

    words_count = df['text'].apply(lambda x: len(get_words(x)))

    chars_count = df['text'].apply(lambda x: len(x))

    

    print("\tFeatures related to words")

    df['capitalized_words_frac'] = df['text'].apply(lambda x: count_capitalized_words(x))/words_count

    df['uppercase_words_frac'] = df['text'].apply(lambda x: count_uppercase_words(x))/words_count

    df['single_frac'] = df['text'].apply(lambda x: count_tokens(x, ['is', 'was', 'has', 'he', 'she', 'it', 'her', 'his']))/words_count

    df['plural_frac'] = df['text'].apply(lambda x: count_tokens(x, ['are', 'were', 'have', 'we', 'they']))/words_count

    

    print("\tFeatures related to chars")

    df['unknown_symb_frac'] = df['text'].apply(lambda x: count_unknown_symbols(x))/chars_count

    df['chars_between_commas_relative'] = df['text'].apply(chars_between_commas)/chars_count   

    

    df['first_word_len_relative'] = df['text'].apply(lambda x: first_word_len(x))/chars_count

    df['last_word_len_relative'] = df['text'].apply(lambda x: last_word_len(x))/chars_count

        

    df['sentiment'] = df['text'].apply(sentiment_nltk)

    df['first_symbol_id'] = df['text'].apply(lambda x: symbol_id(x[0]))

    df['last_symbol_id'] = df['text'].apply(lambda x: symbol_id(x[-1]))
features = ['capitalized_words_frac', 'uppercase_words_frac', 

            'single_frac', 'plural_frac', 'unknown_symb_frac', 

            'chars_between_commas_relative', 'first_word_len_relative', 

            'last_word_len_relative', 'sentiment', 'first_symbol_id', 'last_symbol_id']

_, axes = plt.subplots(4, 3, figsize=(16,16))

for i, feature in enumerate(features):

    sns.violinplot(x='author', y=feature, data=df_train, ax=axes[int(i/3),i%3])

plt.show()