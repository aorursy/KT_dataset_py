# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df1 = pd.read_csv('../input/reddit-selfposts/subreddit_info.csv', delimiter=',',usecols=['subreddit','category_1', 'category_2']).set_index("subreddit")
df1.dataframeName = 'subreddit_info.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.columns
df2 = pd.read_csv('../input/reddit-selfposts/rspct.tsv', delimiter='\t').set_index("subreddit")
df2.shape
df2 = df2.join(df1).drop(["id"],axis=1)
df2.shape
print(df2.category_1.value_counts())
categories_of_interest=['health', 'profession', 'software', 'electronics', 'music', 'sports', 'sex/relationships'
                        , 'hobby', 'geo', 'crypto', 'company/website', 'other', 'anime/manga', 'drugs', 'writing/stories'
                        , 'arts', 'programming', 'autos', 'advice/question', 'education', 'animals', 'social_group'
                        , 'politics/viewpoint', 'food/drink', 'card_game', 'stem', 'hardware/tools', 'religion/supernatural'
                        , 'parenting', 'books',]
subset_df=df2.loc[df2.category_1.isin(categories_of_interest), ['category_1','category_2','title','selftext']]
subset_df.columns = ['category_1','category_2','title','text']
subset_df.head()

data_columns = ['title','text',]
Y_columns = ['category_1','category_2',]

from bs4 import BeautifulSoup
import regex

def preprocess_dataframe(input_df,data_columns,Y_columns):

    df = input_df.loc[:,Y_columns]

    df['text'] = input_df[data_columns].apply(lambda x: ' '.join(x.map(str)), axis=1)
    df['text'] = df['text'].apply( lambda x: BeautifulSoup(str(x),'html.parser').get_text())

    pattern = regex.compile('[\W\d_]+', regex.UNICODE)
    df['text'] = df['text'].apply( lambda x: pattern.sub(' ',str(x)))
    return df

df = preprocess_dataframe(subset_df,data_columns,Y_columns)
df.head()
df = df.sample(frac=1).reset_index(drop=True)
df.head()
df_train = df.iloc[:600000,:]
df_valid = df.iloc[600000:,:]
df_train.shape
df_valid.shape
# we need the class labels encoded into integers for functions in the pipeline
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()

# fit on both, because otherwise we cannot do validation. The actual encoding doesn't play a role in learning.
oe.fit( pd.concat([df_train[Y_columns], df_valid[Y_columns]]).values.reshape(-1, 2)) 
Y_train = oe.transform(df_train[Y_columns].values.reshape(-1, 2))
Y_valid = oe.transform(df_valid[Y_columns].values.reshape(-1, 2))
print('Y training shape', Y_train.shape, Y_train.dtype)
print('Y validation shape', Y_valid.shape, Y_valid.dtype)
from nltk.corpus import stopwords
language_stop_words = stopwords.words('english')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=2) #ngram_range=(1,2)

import numpy as np

#https://stackoverflow.com/a/55742601/4634344
vectorizer.fit(df_train['text'].apply(lambda x: np.str_(x)))
X_train = vectorizer.transform(df_train['text'].apply(lambda x: np.str_(x)))
X_train.shape
X_valid = vectorizer.transform(df_valid['text'].apply(lambda x: np.str_(x)))
X_valid.shape
print('X training shape', X_train.shape, X_train.dtype)
print('X validation shape', X_valid.shape, X_valid.dtype)
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import SGDClassifier

clf=ClassifierChain(SGDClassifier(random_state=0, class_weight='balanced', n_jobs=-1))
from sklearn.metrics import jaccard_score, f1_score, make_scorer

def concat_categories(Y):
  return np.apply_along_axis(lambda a: str(a[0]) + '-' + str(a[1]), 1, Y)

# score for predicting category_1
def js_0(y,y_pred, **kwargs):
  return jaccard_score(y[:,0], y_pred[:,0], average='micro')
# score for predicting category_2
def js_1(y,y_pred, **kwargs):
  return jaccard_score(y[:,1], y_pred[:,1], average='micro')
def f1_0(y,y_pred, **kwargs):
  return f1_score(y[:,0], y_pred[:,0], average='micro')
def f1_1(y,y_pred, **kwargs):
  return f1_score(y[:,1], y_pred[:,1], average='micro')
# score for predicting 'category_1-category_2' (concatenated strings)
def js_01(y,y_pred, **kwargs):
  return jaccard_score(concat_categories(y), concat_categories(y_pred), average='micro')
def f1_01(y,y_pred, **kwargs):
  return f1_score(concat_categories(y), concat_categories(y_pred), average='micro')

js_0_scorer = make_scorer(score_func=js_0, greater_is_better=True, needs_proba=False, needs_threshold=False)
js_1_scorer = make_scorer(score_func=js_1, greater_is_better=True, needs_proba=False, needs_threshold=False)
js_01_scorer = make_scorer(score_func=js_01, greater_is_better=True, needs_proba=False, needs_threshold=False)
f1_0_scorer = make_scorer(score_func=f1_0, greater_is_better=True, needs_proba=False, needs_threshold=False)
f1_1_scorer = make_scorer(score_func=f1_1, greater_is_better=True, needs_proba=False, needs_threshold=False)
f1_01_scorer = make_scorer(score_func=f1_01, greater_is_better=True, needs_proba=False, needs_threshold=False)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_valid)
print('For both Level 1 and Level 2  concatenated:\n\tF1 micro (=accuracy): {}'.format(f1_01(Y_valid,Y_pred).round(3)))
print('Just the Level 1:\n\tF1 micro (=accuracy): {}'.format(f1_0(Y_valid,Y_pred).round(3)))
print('Just the Level 2:\n\tF1 micro (=accuracy): {}'.format(f1_1(Y_valid,Y_pred).round(3)))