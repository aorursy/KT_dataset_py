# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#import required packages
#basics
import pandas as pd 
import numpy as np

#misc
import gc
import time
import warnings

#stats
from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
color = sns.color_palette()

%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


#nlp
import string
import re    #for regex
import nltk
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer   


#FeatureEngineering
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import f1_score

# model
import lightgbm as lgb

eng_stopwords = set(stopwords.words("english"))
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("The shape of train data frame: %s" % str(train_df.shape))
print("The shape of test data frame: %s" % str(test_df.shape))
print("The main columns: %s" % str(train_df.columns.values))
train_df.head()
pos_num = train_df[train_df.target == 1].shape[0]
neg_num = train_df[train_df.target == 0].shape[0]
print("The number of positive %d, and the number of negative %d" % (pos_num, neg_num))
print("The rate of postive %.5f" % (pos_num / train_df.shape[0]))
train_df[train_df.target == 1].question_text.values[:10].tolist()
def generate_indirect_features(df):
    df['count_word'] = df.question_text.apply(lambda x: len(str(x).split()))
    df['count_unique_word']=df.question_text.apply(lambda x: len(set(str(x).split())))
    df['count_letters']=df.question_text.apply(lambda x: len(str(x)))
    df["count_punctuations"] =df.question_text.apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df["count_words_upper"] = df.question_text.apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df["count_words_title"] = df.question_text.apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df["count_stopwords"] = df.question_text.apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
    df["mean_word_len"] = df.question_text.apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['word_unique_percent']=df['count_unique_word']*100/df['count_word']
    df['punct_percent']=df['count_punctuations']*100/df['count_word']
    return df

train_df = generate_indirect_features(train_df)
test_df = generate_indirect_features(test_df)
def violin_chart(df, column_name, min_clip=None, max_clip=None, title=None):
    title = column_name if title is None else title
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=15)
    sub_df = df[[column_name, 'target']]
    min_clip = np.min(sub_df[column_name]) if min_clip is None else min_clip
    max_clip = np.max(sub_df[column_name]) if max_clip is None else max_clip
    sub_df[column_name] = np.clip(df[column_name].values, min_clip, max_clip)
    sns.violinplot(y=column_name, x='target', data=sub_df, split=True, innert="quart")
    plt.xlabel("Is Isincere?", fontsize=12)
    plt.ylabel(column_name, fontsize=12)
    plt.show()
for column_name in train_df.columns.values:
    if column_name not in ['qid', 'question_text', 'target']:
        print(column_name, end=", ")
violin_chart(train_df, "count_word")
violin_chart(train_df, "count_unique_word")
violin_chart(train_df, "count_letters")
violin_chart(train_df, "count_punctuations", max_clip=20)
violin_chart(train_df, "count_words_upper", max_clip=25)
violin_chart(train_df, "count_words_title")
violin_chart(train_df, "count_stopwords")
violin_chart(train_df, "mean_word_len")
violin_chart(train_df, "word_unique_percent")
violin_chart(train_df, "punct_percent", max_clip=70)
tf_idf_vector = TfidfVectorizer(strip_accents='unicode', analyzer='word',ngram_range=(1,1),
                                use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english')
train_vect = tf_idf_vector.fit_transform(train_df.question_text.values)
print("The shape of TF-IDF train matrix: %s" % str(train_vect.shape))
def top_tfidf_words(tfidf_, words, top_n=25):
    topn_ids = np.argsort(tfidf_)[::-1][:top_n]
    top_words = [(words[i], tfidf_[i]) for i in topn_ids]
    df = pd.DataFrame(top_words)
    df.columns = ['word', 'tfidf']
    return df

def top_mean_words(tf_idf_matrix, words, grp_ids, min_tfidf=0.1, top_n=25):
    _matrix = tf_idf_matrix[grp_ids]
#     _matrix[_matrix < min_tfidf] = 0
    tfidf_means = _matrix.mean(axis=0)
    tfidf_means = np.asarray(tfidf_means).reshape(-1)
    return top_tfidf_words(tfidf_means, words, top_n)

def top_words_by_target(tf_idf_matrix, words, min_tfidf=0.1, top_n=20):
    pos_idx = train_df.index[train_df.target == 1].values
    neg_idx = train_df.index[train_df.target == 0].values
    return top_mean_words(tf_idf_matrix, words, pos_idx, min_tfidf, top_n), top_mean_words(tf_idf_matrix, words, neg_idx, min_tfidf, top_n)
pos_top_tfidf, neg_top_tfidf = top_words_by_target(train_vect, tf_idf_vector.get_feature_names())
pos_top_tfidf.head()
trace = go.Bar(
    x = pos_top_tfidf.word,
    y = pos_top_tfidf.tfidf
)
layout = dict(
    title = "Mean TF-IDF of word in positive",
    xaxis = dict(title = 'Word'),
    yaxis = dict(title = 'Mean TF-IDF')
)
data = [trace]

py.iplot(dict(data = data, layout = layout), filename = 'basic-line')
neg_top_tfidf.head()
trace = go.Bar(
    x = neg_top_tfidf.word,
    y = neg_top_tfidf.tfidf
)

layout = dict(
    title = "Mean TF-IDF of word in negative",
    xaxis = dict(title = 'Word'),
    yaxis = dict(title = 'Mean TF-IDF')
)

data = [trace]

py.iplot(dict(data = data, layout = layout), filename = 'basic-line')
tf_idf_vector = TfidfVectorizer(strip_accents='unicode', analyzer='word',ngram_range=(2,2),
                                use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english')
train_vect = tf_idf_vector.fit_transform(train_df.question_text.values)
print("The shape of bi-gram TF-IDF train matrix: %s" % str(train_vect.shape))
pos_top_tfidf, neg_top_tfidf = top_words_by_target(train_vect, tf_idf_vector.get_feature_names())
pos_top_tfidf.head()
neg_top_tfidf.head()
trace = go.Bar(
    x = pos_top_tfidf.word,
    y = pos_top_tfidf.tfidf
)

layout = dict(
    title = "Mean TF-IDF of word in positive (Bigram)",
    xaxis = dict(title = 'Word'),
    yaxis = dict(title = 'Mean TF-IDF')
)

data = [trace]

py.iplot(dict(data = data, layout = layout), filename = 'basic-line')
trace = go.Bar(
    x = neg_top_tfidf.word,
    y = neg_top_tfidf.tfidf
)

layout = dict(
    title = "Mean TF-IDF of word in negative (Bigram)",
    xaxis = dict(title = 'Word'),
    yaxis = dict(title = 'Mean TF-IDF')
)

data = [trace]

py.iplot(dict(data = data, layout = layout), filename = 'basic-line')
count_vector = CountVectorizer(strip_accents='unicode', analyzer='word',ngram_range=(1,1),  stop_words = 'english')
train_vect = count_vector.fit_transform(train_df.question_text.values)
print("The shape of count train matrix: %s" % str(train_vect.shape))
pos_top_count, neg_top_count = top_words_by_target(train_vect, count_vector.get_feature_names())
pos_top_count.head()
trace = go.Bar(
    x = pos_top_count.word,
    y = pos_top_count.tfidf
)

layout = dict(
    title = "Mean Count Rate of word in positive (Bigram)",
    xaxis = dict(title = 'Word'),
    yaxis = dict(title = 'Mean Count Rate')
)

data = [trace]

py.iplot(dict(data = data, layout = layout), filename = 'basic-line')
neg_top_count.head()
trace = go.Bar(
    x = neg_top_count.word,
    y = neg_top_count.tfidf
)

layout = dict(
    title = "Mean Count Rate of word in negative (Bigram)",
    xaxis = dict(title = 'Word'),
    yaxis = dict(title = 'Mean Count Rate')
)

data = [trace]

py.iplot(dict(data = data, layout = layout), filename = 'basic-line')
def count_keywords(df, word):
    df["count_%s" % word] = df.question_text.apply(lambda x: x.lower().count(word))
train_df['count_muslim'] = train_df.question_text.apply(lambda x: x.lower().count("muslim"))
test_df['count_muslim'] = test_df.question_text.apply(lambda x: x.lower().count("muslim"))
violin_chart(train_df, "count_muslim")
keywords = ["trump", "chinese people", "black", "white people", "indians", "muslims", "sex", "india"]
for keyword in keywords:
    count_keywords(train_df, keyword)
    count_keywords(test_df, keyword)
tf_idf_word_vector = TfidfVectorizer(strip_accents='unicode', analyzer='word',ngram_range=(1, 2), 
                                     use_idf=1, smooth_idf=1, sublinear_tf=1, stop_words = 'english', 
                                     max_features=200000)

# tf_idf_char_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', 
#                                      analyzer='char', token_pattern=r'\w{1,}',stop_words='english',
#                                      ngram_range=(2, 5), max_features=50000)
# tf_idf_char_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='char_wb', token_pattern=r'\w{1,}',
#                                      stop_words='english', ngram_range=(2, 5), max_features=50000)

train_word_tfidf = tf_idf_word_vector.fit_transform(train_df.question_text)
# train_char_tfidf = tf_idf_char_vector.fit_transform(train_df.question_text)
# train_tfidf = sparse.hstack([train_word_tfidf, train_char_tfidf]).tocsr()
train_tfidf = train_word_tfidf
del train_word_tfidf
# del train_char_tfidf
gc.collect()
print("The train tf-idf shape %s" % str(train_tfidf.shape))

test_word_tfidf = tf_idf_word_vector.transform(test_df.question_text)
# test_char_tfidf = tf_idf_char_vector.transform(test_df.question_text)
# test_tfidf = sparse.hstack([test_word_tfidf, test_char_tfidf]).tocsr()
test_tfidf = test_word_tfidf
del test_word_tfidf
# del test_char_tfidf
gc.collect()

print("The test tf-idf shape %s" % str(test_tfidf.shape))
indirect_features_name = [feat_name for feat_name in train_df.columns.values if feat_name not in ["qid", "question_text", "target"]]
# indirect_features_name
train_indirect_features = train_df[indirect_features_name].values
test_indirect_features = test_df[indirect_features_name].values
target = train_df.target.values

# prepare to delete the train dataframe and test dataframe
num_train = train_df.shape[0]
submission_df = pd.DataFrame({"qid":test_df["qid"].values})
validation_df = pd.DataFrame({"qid":train_df["qid"].values})
del train_df
del test_df
gc.collect()

X_train = sparse.hstack([train_tfidf, train_indirect_features]).tocsr()
X_test = sparse.hstack([test_tfidf, test_indirect_features]).tocsr()
del train_tfidf
del test_tfidf
gc.collect()
def lgb_f1_score(y_pre, data):
    y_true = data.get_label()
    best_f1 = f1_score(y_true, (y_pre>0.5).astype(int))
#     best_f1 = 0
#     for thresh in np.arange(0.1, 0.501, 0.01):
#         thresh = np.round(thresh, 2)
#         _f1 = f1_score(y_true, (y_pre>thresh).astype(int))
#         if _f1 > best_f1:
#             best_f1 = _f1
    return 'f1', best_f1, True
# Set LGBM parameters
params = {
    "objective": "binary",
    'metric': {'auc'},
    "boosting_type": "gbdt",
    "verbosity": -1,
    "num_threads": 4,
    "bagging_fraction": 0.8,
    "feature_fraction": 0.8,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "verbose": -1,
    "min_split_gain": .1,
    "reg_alpha": .1,
    "device_type": "gpu",
    "seed": 2018
}

scores = []
folds = KFold(n_splits=5)
indices = np.arange(num_train)
trn_lgbset = lgb.Dataset(data=X_train, label=target, free_raw_data=False)
valid_predict = np.zeros(num_train, dtype=np.float32)
mean_best_iter = 0
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(indices)):
    print("valid in the %d fold" % (n_fold + 1))
    model = lgb.train(
        params=params,
        train_set=trn_lgbset.subset(trn_idx),
        num_boost_round=1000,
        valid_sets=[trn_lgbset.subset(val_idx)],
        early_stopping_rounds=50,
#         feval=lgb_f1_score,
        verbose_eval=200
    )
    mean_best_iter += model.best_iteration / 5
    valid_predict[val_idx] = model.predict(trn_lgbset.data[val_idx], num_iteration=model.best_iteration)
best_thresh = 0
best_f1 = 0
for thresh in np.arange(0.1, 0.501, 0.01):
    thresh = np.round(thresh, 2)
    _f1 = f1_score(target, (valid_predict>thresh).astype(int))
    if _f1 > best_f1:
        best_f1 = _f1
        best_thresh = thresh
    print("\tF1 score at threshold {0} is {1}".format(thresh, _f1))

print("Best F1 score {0}, Best thresh {1}".format(best_f1, best_thresh))
model = lgb.train(
    params=params,
    train_set=trn_lgbset,
    feval=lgb_f1_score,
    num_boost_round=int(mean_best_iter)
)

predict = model.predict(X_test, num_iteration=model.best_iteration)
validation_df["prediction"] = valid_predict
validation_df.to_csv("validation.csv", index=False)

submission_df["prediction"] = (predict > best_thresh).astype(int)
submission_df.to_csv("submission.csv", index=False)