import numpy as np

import pandas as pd

pd.set_option("display.max_colwidth", 80)

import os

import matplotlib.pyplot as plt

import seaborn as sns

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import modeling_functions as mf

import nlp_preprocessing_functions as npf

import preprocessing_functions as pf

import holoviews as hv

from holoviews import opts

hv.extension('bokeh')

from wordcloud import STOPWORDS

from collections import defaultdict

import string
train = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/train.csv")

print('Train Set Shape = {}'.format(train.shape))

train.head()
test = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/test.csv")

print('Test Set Shape = {}'.format(test.shape))

test.head()
submission = pd.read_csv("/kaggle/input/quora-insincere-questions-classification/sample_submission.csv")

submission.head()
print(f'the number of insincere questions is : {len(train[train["target"]==1])} / {len(train)}')

print(f'the number of not-insincere questions is : {len(train[train["target"]==0])} / {len(train)}')

train_counts = train["target"].value_counts()

hv.Bars((train_counts.keys(), train_counts.values),"Target Label","Counts").opts(width=600,height=400,title="Target Counts",tools=['hover'])
print(f'the number of nulls in train set : {train.isnull().any().sum()}')

print(f'the number of nulls in test set : {test.isnull().any().sum()}')
#ngram function

def generate_ngrams(text, n_gram=1):

    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]

    ngrams = zip(*[token[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]
train_0 = train[train["target"]==0]

train_1 = train[train["target"]==1]



freq_dict_0_uni = defaultdict(int)

for sent in train_0["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict_0_uni[word] += 1



freq_dict_1_uni = defaultdict(int)

for sent in train_1["question_text"]:

    for word in generate_ngrams(sent):

        freq_dict_1_uni[word] += 1
data_0_uni = list(sorted(freq_dict_0_uni.items(), key=lambda x: x[1],reverse=True))

bars_0_uni = hv.Bars(data_0_uni[0:50][::-1],"Word","Count").opts(invert_axes=True, width=500, height=800, title="unigram frequency with target=0", color="red")

data_1_uni = list(sorted(freq_dict_1_uni.items(), key=lambda x: x[1],reverse=True))

bars_1_uni = hv.Bars(data_1_uni[0:50][::-1],"Word","Count").opts(invert_axes=True, width=500, height=800, title="unigram frequency with target=1", color="blue")



(bars_0_uni + bars_1_uni).opts(opts.Bars(tools=['hover']))
freq_dict_0_bi = defaultdict(int)

for sent in train_0["question_text"]:

    for word in generate_ngrams(sent,n_gram=2):

        freq_dict_0_bi[word] += 1



freq_dict_1_bi = defaultdict(int)

for sent in train_1["question_text"]:

    for word in generate_ngrams(sent,n_gram=2):

        freq_dict_1_bi[word] += 1
data_0_bi = list(sorted(freq_dict_0_bi.items(), key=lambda x: x[1],reverse=True))

bars_0_bi = hv.Bars(data_0_bi[0:50][::-1],"Word","Count").opts(invert_axes=True, width=500, height=800, title="bigram frequency with target=0", color="red")

data_1_bi = list(sorted(freq_dict_1_bi.items(), key=lambda x: x[1],reverse=True))

bars_1_bi = hv.Bars(data_1_bi[0:50][::-1],"Word","Count").opts(invert_axes=True, width=500, height=800, title="bigram frequency with target=1", color="blue")



(bars_0_bi + bars_1_bi).opts(opts.Bars(tools=['hover']))
freq_dict_0_tri = defaultdict(int)

for sent in train_0["question_text"]:

    for word in generate_ngrams(sent,n_gram=3):

        freq_dict_0_tri[word] += 1



freq_dict_1_tri = defaultdict(int)

for sent in train_1["question_text"]:

    for word in generate_ngrams(sent,n_gram=3):

        freq_dict_1_tri[word] += 1
data_0_tri = list(sorted(freq_dict_0_tri.items(), key=lambda x: x[1],reverse=True))

bars_0_tri = hv.Bars(data_0_tri[0:50][::-1],"Word","Count").opts(invert_axes=True, width=500, height=800, title="trigram frequency with target=0", color="red")

data_1_tri = list(sorted(freq_dict_1_tri.items(), key=lambda x: x[1],reverse=True))

bars_1_tri = hv.Bars(data_1_tri[0:50][::-1],"Word","Count").opts(invert_axes=True, width=500, height=800, title="trigram frequency with target=1", color="blue")



(bars_0_tri + bars_1_tri).opts(opts.Bars(tools=['hover']))
# sentences_raw_train = train['question_text']

# sentences_raw_test = test['question_text']

# sentences_preprocessed_train = []

# sentences_preprocessed_test = []

# for sentence in sentences_raw_train:

#     lemmas = npf.tokenizer(sentence)

#     sentence_without_stop_words = npf.remove_stop_words(lemmas)

#     sentences_preprocessed_train.append(sentence_without_stop_words)

# sentences_preprocessed_train = [" ".join(doc) for doc in sentences_preprocessed_train]

# for sentence in sentences_raw_test:

#     lemmas = npf.tokenizer(sentence)

#     sentence_without_stop_words = npf.remove_stop_words(lemmas)

#     sentences_preprocessed_test.append(sentence_without_stop_words)

# sentences_preprocessed_test = [" ".join(doc) for doc in sentences_preprocessed_test]

# sentences_tfidf = copy.copy(sentences_preprocessed_train)

# sentences_tfidf.extend(sentences_preprocessed_test)
# tfidf_train, tfidf_test = npf.tfidf_features(docs_tfidf=sentences_tfidf, docs_train=sentences_preprocessed_train, docs_test=sentences_preprocessed_test, _max_features=1000)

# tfidf_train.head()
# tfidf_test.head()
uni_0_words = [word_freq[0] for word_freq in data_0_uni[0:50]]

uni_1_words = [word_freq[0] for word_freq in data_1_uni[0:50]]

bi_0_words = [word_freq[0] for word_freq in data_0_bi[0:50]]

bi_1_words = [word_freq[0] for word_freq in data_1_bi[0:50]]

tri_0_words = [word_freq[0] for word_freq in data_0_tri[0:50]]

tri_1_words = [word_freq[0] for word_freq in data_1_tri[0:50]]
for df in [train,test]:

    uni_0_feature = []

    uni_1_feature = []

    bi_0_feature = []

    bi_1_feature = []

    tri_0_feature = []

    tri_1_feature = []

    for line in df["question_text"]:

        uni_0_len = len([word for word in uni_0_words if word in line])

        uni_1_len = len([word for word in uni_1_words if word in line])

        bi_0_len = len([word for word in bi_0_words if word in line])

        bi_1_len = len([word for word in bi_1_words if word in line])

        tri_0_len = len([word for word in tri_0_words if word in line])

        tri_1_len = len([word for word in tri_1_words if word in line])

        

        uni_0_feature.append(uni_0_len)

        uni_1_feature.append(uni_1_len)

        bi_0_feature.append(bi_0_len)

        bi_1_feature.append(bi_1_len)

        tri_0_feature.append(tri_0_len)

        tri_1_feature.append(tri_1_len)

    df["uni_0"] = uni_0_feature

    df["uni_1"] = uni_1_feature

    df["bi_0"] = bi_0_feature

    df["bi_1"] = bi_1_feature

    df["tri_0"] = tri_0_feature

    df["tri_1"] = tri_1_feature

        
train.head()
test.head()
for df in [train,test]:

    df["sent_len"] = df["question_text"].apply(lambda x: len(x.split()))

    df["word_mean_len"] = df["question_text"].apply(lambda x: np.mean([len(i) for i in x.split()]))

    df["punc_num"] = df["question_text"].apply(lambda x: len([c for c in x.split() if c in string.punctuation]))
train.head()
test.head()