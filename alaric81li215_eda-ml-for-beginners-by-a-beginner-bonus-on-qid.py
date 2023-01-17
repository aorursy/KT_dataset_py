# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
import string
import pickle
import random

import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk

import warnings
warnings.filterwarnings('ignore')

stop_words = set(nltk.corpus.stopwords.words('english')) 

sns.set()
filepath_train = os.path.join('..', 'input', 'train.csv')
filepath_test = os.path.join('..', 'input', 'test.csv')
df_train = pd.read_csv(filepath_train)
df_test = pd.read_csv(filepath_test)
df_train.shape, df_test.shape
df_train.head()
df_train.info()
# Separating the targets from the feature we will work on
X = df_train.drop(['qid', 'target'], axis=1)
y = df_train['target']
X.shape, y.shape
n_0 = y.value_counts()[0]
n_1 = y.value_counts()[1]
print('{}% of the questions in the train set are tagged as insincere.'.format((n_1*100/(n_1 + n_0)).round(2)))
# Visualizing some insincere questions randomly chosen

np.array(X[y==1])[np.random.choice(len(np.array(X[y==1])), size=15, replace=False)]

# Custom function to create the meta-features we want from X and add them in a new DataFrame

def add_metafeatures(dataframe):
    new_dataframe = dataframe.copy()
    questions = df_train['question_text']
    n_charac = pd.Series([len(t) for t in questions])
    n_punctuation = pd.Series([sum([1 for x in text if x in set(string.punctuation)]) for text in questions])
    n_upper = pd.Series([sum([1 for c in text if c.isupper()]) for text in questions])
    new_dataframe['n_charac'] = n_charac
    new_dataframe['n_punctuation'] = n_punctuation
    new_dataframe['n_upper'] = n_upper
    return new_dataframe
X_meta = add_metafeatures(X)
X_meta.head()
print('Number of characters description : \n\n {} \n\n Number of punctuations description : \n\n {} \n\n Number of uppercase characters description : \n\n {}'.format(
    X_meta['n_charac'].describe(),
    X_meta['n_punctuation'].describe(), 
    X_meta['n_upper'].describe()))
# Separating X_meta with our targets in y

X_meta_sincere = X_meta[y==0]
X_meta_insincere = X_meta[y==1]
_, axes = plt.subplots(2, 3, sharey=True, figsize=(18, 8))
sns.boxplot(x=X_meta['n_charac'], y=y, orient='h', ax=axes.flat[0]);
sns.boxplot(x=X_meta['n_punctuation'], y=y, orient='h', ax=axes.flat[1]);
sns.boxplot(x=X_meta['n_upper'], y=y, orient='h', ax=axes.flat[2]);

X_meta_charac = X_meta[X_meta['n_charac']<400]
X_meta_punctuation = X_meta[X_meta['n_punctuation']<10]
X_meta_upper = X_meta[X_meta['n_upper']<15]

sns.boxplot(x=X_meta_charac['n_charac'], y=y, orient='h', ax=axes.flat[3]);
sns.boxplot(x=X_meta_punctuation['n_punctuation'], y=y, orient='h', ax=axes.flat[4]);
sns.boxplot(x=X_meta_upper['n_upper'], y=y, orient='h', ax=axes.flat[5]);
pd.concat([X_meta[X_meta['n_charac']>400], y], axis=1, join='inner')
print(np.array(X_meta[X_meta['n_charac']>400]['question_text']))
punctuation_ratio = 100*X_meta['n_punctuation'] / X_meta['n_charac']
plt.figure(figsize=(18, 8))
sns.boxplot(punctuation_ratio, y, orient='h');
pd.concat([X_meta[punctuation_ratio>50], y, punctuation_ratio], axis=1, join='inner')
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
vectorizer = CountVectorizer(stop_words='english')
svd = TruncatedSVD(n_components=1, random_state=42)
preprocessing_pipe = Pipeline([('vectorizer', vectorizer), ('svd', svd)])
# Building the latent semantic analysis dataframe for sincere and insincere questions

lsa_insincere = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere = pd.DataFrame(svd.components_)
topics_insincere.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()

lsa_sincere = preprocessing_pipe.fit_transform(X[y==0]['question_text'])
topics_sincere = pd.DataFrame(svd.components_)
topics_sincere.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()

topics_insincere.shape, topics_sincere.shape
fig, axes = plt.subplots(1, 2, figsize=(22,10));

topics_sincere.iloc[0].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=axes[0]);
topics_insincere.iloc[0].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=axes[1]);
vectorizer = CountVectorizer(stop_words='english')
svd = TruncatedSVD(n_components=2, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer', vectorizer), ('svd', svd)])

# Building the latent semantic analysis dataframe for sincere and insincere questions

lsa_insincere_2 = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere_2 = pd.DataFrame(svd.components_)
topics_insincere_2.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()

lsa_sincere_2 = preprocessing_pipe.fit_transform(X[y==0]['question_text'])
topics_sincere_2 = pd.DataFrame(svd.components_)
topics_sincere_2.columns = preprocessing_pipe.named_steps['vectorizer'].get_feature_names()


fig_1, axes_1 = plt.subplots(1, 2, figsize=(18, 8))
for i, ax in enumerate(axes_1.flat):
    topics_insincere_2.iloc[i].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=ax)
    
fig_2, axes_2 = plt.subplots(1, 2, figsize=(18, 8))
for i, ax in enumerate(axes_2.flat):
    topics_sincere_2.iloc[i].sort_values(ascending=False)[:30].sort_values().plot.barh(ax=ax)
vectorizer_22 = CountVectorizer(stop_words='english', ngram_range=(2, 2))
svd_10c = TruncatedSVD(n_components=9, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer_22', vectorizer_22), ('svd_10c', svd_10c)])

# Building the latent semantic analysis dataframe for insincere questions

lsa_insincere_10c = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere_10c = pd.DataFrame(svd_10c.components_)
topics_insincere_10c.columns = preprocessing_pipe.named_steps['vectorizer_22'].get_feature_names()

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
for i, ax in enumerate(axes.flat):
    topics_insincere_10c.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)
vectorizer_23 = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
svd_9c = TruncatedSVD(n_components=9, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer_23', vectorizer_23), ('svd_9c', svd_9c)])

# Building the latent semantic analysis dataframe for insincere questions

lsa_insincere_9c = preprocessing_pipe.fit_transform(X[y==1]['question_text'])
topics_insincere_9c = pd.DataFrame(svd_9c.components_)
topics_insincere_9c.columns = preprocessing_pipe.named_steps['vectorizer_23'].get_feature_names()

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
for i, ax in enumerate(axes.flat):
    topics_insincere_9c.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
#Custom function to preprocess the questions

def preprocess(X):
    docs = nlp.pipe(X)
    lemmas_as_string = []
    for doc in docs:
        doc_of_lemmas = []
        for t in doc:
            if t.text.lower() not in stop_words and t.text.isalpha() == True:
                if t.lemma_ !='-PRON-':
                    doc_of_lemmas.append(t.lemma_)
                else:
                    doc_of_lemmas.append(t.text)
        lemmas_as_string.append(' '.join(doc_of_lemmas))
    return pd.DataFrame(lemmas_as_string)
%%time
X_prep = preprocess(X['question_text'])
X_prep.to_pickle('X_preprocessed.pkl')
X_prep = pd.read_pickle('X_preprocessed.pkl')
X_prep.columns = ['question_text']
X_prep.head()
vectorizer_23 = TfidfVectorizer(stop_words='english', ngram_range=(2, 3))
svd_9c = TruncatedSVD(n_components=9, random_state=42)

preprocessing_pipe = Pipeline([('vectorizer_23', vectorizer_23), ('svd_9c', svd_9c)])

# Building the latent semantic analysis dataframe for insincere questions

lsa_insincere_9c = preprocessing_pipe.fit_transform(X_prep[y==1]['question_text'])
topics_insincere_9c = pd.DataFrame(svd_9c.components_)
topics_insincere_9c.columns = preprocessing_pipe.named_steps['vectorizer_23'].get_feature_names()

fig, axes = plt.subplots(3, 3, figsize=(20, 12))
for i, ax in enumerate(axes.flat):
    topics_insincere_9c.iloc[i].sort_values(ascending=False)[:10].sort_values().plot.barh(ax=ax)
df_train_qid = df_train.copy()
df_train_qid['qid_base_ten'] = df_train_qid['qid'].apply(lambda x : int(x, 16))
df_train_qid.head()
min_qid = df_train_qid['qid_base_ten'].min()
max_qid = df_train_qid['qid_base_ten'].max()
df_train_qid['qid_base_ten_normalized'] = df_train_qid['qid_base_ten'].apply(lambda x : (x - min_qid)/min_qid)
plt.figure(figsize=(18, 8));
plt.scatter(x=df_train_qid['qid_base_ten_normalized'][:100], y=df_train_qid.index[:100]);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index in df_train_qid');
df_test_qid = df_test.copy()

df_test_qid['qid_base_ten'] = df_test_qid['qid'].apply(lambda x : int(x, 16))

df_test_qid['qid_base_ten_normalized'] = df_test_qid['qid_base_ten'].apply(lambda x : (x - min_qid)/min_qid)

plt.figure(figsize=(18, 8));
plt.scatter(x=df_test_qid['qid_base_ten_normalized'][:100], y=df_test_qid.index[:100]);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index in df_test_qid');
df_train_qid.drop('target', axis=1, inplace=True)
df_train_qid['test_or_train'] = 'train'
df_test_qid['test_or_train'] = 'test'
df_qid = pd.concat([df_train_qid, df_test_qid]).sort_values('qid_base_ten_normalized').reset_index()
df_qid.drop('index', axis=1, inplace=True)
df_qid.head()
df_qid_train = df_qid[df_qid['test_or_train']=='train']
df_qid_test = df_qid[df_qid['test_or_train']=='test']

plt.figure(figsize=(18, 8));
plt.scatter(x=df_qid_train['qid_base_ten_normalized'], y=df_qid_train.index, label='Train');
plt.scatter(x=df_qid_test['qid_base_ten_normalized'], y=df_qid_test.index, label='Test',s=5);
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index');
plt.title('qid_base_ten_normalized for train and test datasets')
plt.legend();
plt.figure(figsize=(18, 8));
plt.scatter(x=df_qid_train['qid_base_ten_normalized'][:1500], y=df_qid_train.index[:1500], label='Train');
plt.scatter(x=df_qid_test['qid_base_ten_normalized'][:50], y=df_qid_test.index[:50], label='Test',s=150, marker='d');
plt.xlabel('qid_base_ten_normalized');
plt.ylabel('Question index');
plt.title('qid_base_ten_normalized for the first 1500 train points and 50 test points')
plt.legend();
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X['question_text'], y, test_size=.2, random_state=42, stratify=y)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
lr = LogisticRegression()
pipe_baseline = Pipeline([('tfidf', tfidf), ('lr', lr)])
pipe_baseline.fit(X_train, y_train)
y_pred = pipe_baseline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

ax = plt.gca()
sns.heatmap(cm, cmap='Blues', cbar=False, annot=True, xticklabels=y_test.unique(), yticklabels=y_test.unique(), ax=ax);
ax.set_xlabel('y_pred');
ax.set_ylabel('y_true');
ax.set_title('Confusion Matrix');
cr = classification_report(y_test, y_pred)
print(cr)
y_prob = pipe_baseline.predict_proba(X_test)
best_threshold = 0
f1=0
for i in np.arange(.1, .51, 0.01):
    y_pred = [1 if proba>i else 0 for proba in y_prob[:, 1]]
    f1score = f1_score(y_pred, y_test)
    if f1score>f1:
        best_threshold = i
        f1=f1score
        
y_pred = [1 if proba>best_threshold else 0 for proba in y_prob[:, 1]]
f1 = f1_score(y_pred, y_test)
print('The best threshold is {}, with an f1_score of {}'.format(best_threshold, f1))
