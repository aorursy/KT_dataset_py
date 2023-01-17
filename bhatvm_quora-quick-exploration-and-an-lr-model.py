import os
import json
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import *
from wordcloud import *
from tqdm import tqdm

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split
from sklearn import metrics

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


import os
print(os.listdir("../input"))
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)
train_df.head()
train_df['target'].value_counts().plot(kind='bar')
target_table = pd.crosstab(index = train_df['target'], columns='count')
print(target_table/target_table.sum())
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]
train0_df = train_df[train_df['target']==0]
train1_df = train_df[train_df['target']==1]

#For all sincere questions
from collections import defaultdict
sinc_dict = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent):
        sinc_dict[word] += 1
sinc_sorted = pd.DataFrame(sorted(sinc_dict.items(), key=lambda x: x[1])[::-1])
sinc_sorted.columns = ["word", "wordcount"]

#For all insincere questions
from collections import defaultdict
insinc_dict = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent):
        insinc_dict[word] += 1
insinc_sorted = pd.DataFrame(sorted(insinc_dict.items(), key=lambda x: x[1])[::-1])
insinc_sorted.columns = ["word", "wordcount"]
gram1_0 = go.Bar(y = sinc_sorted["word"].head(20),x = sinc_sorted["wordcount"].head(20),orientation="h")
gram1_1 = go.Bar(y = insinc_sorted["word"].head(20),x = insinc_sorted["wordcount"].head(20),orientation="h")

fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig.append_trace(gram1_0, 1, 1)
fig.append_trace(gram1_1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')
sinc_dict2 = defaultdict(int)
for sent in train0_df["question_text"]:
    for word in generate_ngrams(sent,n_gram=2):
        sinc_dict2[word] += 1
sinc_sorted_2 = pd.DataFrame(sorted(sinc_dict2.items(), key=lambda x: x[1])[::-1])
sinc_sorted_2.columns = ["word", "wordcount"]

#For all insincere questions
from collections import defaultdict
insinc_dict_2 = defaultdict(int)
for sent in train1_df["question_text"]:
    for word in generate_ngrams(sent,n_gram=2):
        insinc_dict_2[word] += 1
insinc_sorted_2 = pd.DataFrame(sorted(insinc_dict_2.items(), key=lambda x: x[1])[::-1])
insinc_sorted_2.columns = ["word", "wordcount"]
gram2_0 = go.Bar(y = sinc_sorted_2["word"].head(20),x = sinc_sorted_2["wordcount"].head(20),orientation="h")
gram2_1 = go.Bar(y = insinc_sorted_2["word"].head(20),x = insinc_sorted_2["wordcount"].head(20),orientation="h")

fig2 = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of sincere questions", 
                                          "Frequent words of insincere questions"])
fig2.append_trace(gram2_0, 1, 1)
fig2.append_trace(gram2_1, 1, 2)
fig2['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig2, filename='word-plots')
from wordcloud import WordCloud, STOPWORDS

## Number of words in the text ##
train_df["num_words"] = train_df["question_text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["question_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["question_text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["question_text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##
train_df["num_chars"] = train_df["question_text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["question_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
test_df["num_stopwords"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['question_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["question_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
train_df.head()
## Truncate some extreme values for better visuals ##
train_df['num_words'].loc[train_df['num_words']>60] = 60 #truncation for better visuals
train_df['num_punctuations'].loc[train_df['num_punctuations']>10] = 10 #truncation for better visuals
train_df['num_chars'].loc[train_df['num_chars']>350] = 350 #truncation for better visuals

f, axes = plt.subplots(3, 1, figsize=(10,20))
sns.boxplot(x='target', y='num_words', data=train_df, ax=axes[0])
axes[0].set_xlabel('Target', fontsize=12)
axes[0].set_title("Number of words in each class", fontsize=15)

sns.boxplot(x='target', y='num_chars', data=train_df, ax=axes[1])
axes[1].set_xlabel('Target', fontsize=12)
axes[1].set_title("Number of characters in each class", fontsize=15)

sns.boxplot(x='target', y='num_punctuations', data=train_df, ax=axes[2])
axes[2].set_xlabel('Target', fontsize=12)
#plt.ylabel('Number of punctuations in text', fontsize=12)
axes[2].set_title("Number of punctuations in each class", fontsize=15)
plt.show()
#def tokenize(data):
#    tokenized_docs = [word_tokenize(doc.lower()) for doc in data]
#    alpha_tokens = [[t for t in doc if t.isalpha() == True] for doc in tokenized_docs]
#    stemmer = PorterStemmer ()
#    stemmed_tokens = [[stemmer.stem(alpha) for alpha in doc] for doc in alpha_tokens]
#    X_stem_as_string = [" ".join(x_t) for x_t in stemmed_tokens]
#    return X_stem_as_string
X = train_df['question_text']
y = train_df['target']
X_test = test_df['question_text']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_val.shape, y_val.shape
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
rf = ensemble.RandomForestClassifier(class_weight='balanced_subsample')
tfvec = TfidfVectorizer(stop_words='english', lowercase=False)
pipe = Pipeline([
    ('vectorizer', tfvec),
    ('rf', rf )
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_val)
cm = metrics.confusion_matrix(y_val, y_pred)

ax = plt.gca()
sns.heatmap(cm, cmap='Blues', cbar=False, annot=True, xticklabels=y_val.unique(), yticklabels=y_val.unique(), ax=ax);
ax.set_xlabel('y_pred');
ax.set_ylabel('y_true');
ax.set_title('Confusion Matrix');

cr = metrics.classification_report(y_val, y_pred)
print(cr)
lr = linear_model.LogisticRegression()
pipe_lr = Pipeline([
    ('vectorizer', tfvec),
    ('lr', lr )
])
pipe_lr.fit(X_train, y_train)
y_pred_lr = pipe_lr.predict(X_val)
cm_lr = metrics.confusion_matrix(y_val, y_pred_lr)

ax = plt.gca()
sns.heatmap(cm_lr, cmap='Blues', cbar=False, annot=True, xticklabels=y_val.unique(), yticklabels=y_val.unique(), ax=ax);
ax.set_xlabel('y_pred');
ax.set_ylabel('y_true');
ax.set_title('Confusion Matrix');

cr = metrics.classification_report(y_val, y_pred_lr)
print(cr)
metrics.f1_score(y_pred=y_pred_lr,y_true=y_val)
y_prob_lr = pipe_lr.predict_proba(X_val)
best_threshold = 0
f1=0
for i in np.arange(.1, .51, 0.01):
    y_pred2_lr = [1 if proba>i else 0 for proba in y_prob_lr[:, 1]]
    f1score = metrics.f1_score(y_pred=y_pred2_lr, y_true=y_val)
    if f1score>f1:
        best_threshold = i
        f1=f1score
        
y_pred2_lr = [1 if proba>best_threshold else 0 for proba in y_prob_lr[:, 1]]
f1 = metrics.f1_score(y_pred2_lr, y_val)
print('The best threshold is {}, with an f1_score of {}'.format(best_threshold, f1))
y_pred_sub = pipe_lr.predict(X_test) 

sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = (y_pred_sub > best_threshold).astype(int)
sub.to_csv("submission.csv", index=False)
lr = linear_model.LogisticRegression(penalty='l2',solver='sag')
pipe_cv = Pipeline([
    ('vectorizer', tfvec),
    ('lr', lr )
])
param_grid = {'lr__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

clf = model_selection.GridSearchCV(pipe_cv, param_grid,cv=5)
clf.fit(X_train,y_train)
y_pred_lrcv = clf.best_estimator_.predict(X_val)
print(metrics.f1_score(y_val, y_pred_lrcv))
print(metrics.classification_report(y_val, y_pred_lrcv))
### Lets find the best threshold for cut-off
y_prob_lrcv = clf.best_estimator_.predict_proba(X_val)
best_threshold = 0
f1=0
for i in np.arange(.1, .51, 0.01):
    y_pred2_lrcv = [1 if proba>i else 0 for proba in y_prob_lrcv[:, 1]]
    f1score = metrics.f1_score(y_pred=y_pred2_lrcv, y_true=y_val)
    if f1score>f1:
        best_threshold = i
        f1=f1score
        
y_pred2_lrcv = [1 if proba>best_threshold else 0 for proba in y_prob_lrcv[:, 1]]
f1 = metrics.f1_score(y_pred2_lrcv, y_val)
print('The best threshold is {}, with an f1_score of {}'.format(best_threshold, f1))
y_pred_sub = pipe_lr.predict(X_test) 

sub = pd.read_csv('../input/sample_submission.csv')
sub.prediction = (y_pred_sub > best_threshold).astype(int)
sub.to_csv("submission.csv", index=False)
