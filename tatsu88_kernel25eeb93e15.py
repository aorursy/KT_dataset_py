# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#contractionsモジュールのインストール

!pip install contractions
#NLP

import string #記号(punctuation)の一覧を取得

import re

import contractions #文書の短縮形を修正

from wordcloud import STOPWORDS #ストップワードのリストを取得

from collections import defaultdict #n-gram

from sklearn.feature_extraction.text import CountVectorizer #特徴抽出
# 訓練データとテストデータの読み込み

df_train = pd.read_csv('../input/nlp-getting-started/train.csv', dtype={'id': np.int16, 'target': np.int8})

df_test = pd.read_csv('../input/nlp-getting-started/test.csv', dtype={'id': np.int16})



# 訓練データとテストデータの行数・列数を表示する

print('Training Set Shape = {}'.format(df_train.shape))

print('Test Set Shape = {}'.format(df_test.shape))



# 訓練データの中から10行をランダムで抽出

df_train.sample(n=10, random_state=28)
def fix_contractions(text):

    return contractions.fix(text)



# 関数適応前のツイート例

print("tweet before contractions fix : ", df_train.iloc[1055]["text"])



# 関数の適用

df_train['text']=df_train['text'].apply(lambda x : fix_contractions(x))

df_test['text']=df_test['text'].apply(lambda x : fix_contractions(x))



# 関数適用後のツイート例

print("tweet after contractions fix : ", df_train.iloc[1055]["text"])
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



# 関数適応前のツイート例

print("tweet before URL removal : ", df_train.iloc[1055]["text"])



# 関数の適用

df_train['text']=df_train['text'].apply(lambda x : remove_URL(x))

df_test['text']=df_test['text'].apply(lambda x : remove_URL(x))



# 関数適用後のツイート例

print("tweet after URL removal : ", df_train.iloc[1055]["text"])
def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



# 関数適応前のツイート例

print("tweet before punctuation removal : ", df_train.iloc[1055]["text"])



# 関数の適用

df_train['text']=df_train['text'].apply(lambda x : remove_punct(x))

df_test['text']=df_test['text'].apply(lambda x : remove_punct(x))



# 関数適用後のツイート例

print("tweet after punctuation removal : ", df_train.iloc[1055]["text"])
df_test['text']
stop_words = []

token_pattern = r'[^\s]+'

ngram_range = (1, 2)

max_df = 1.0

min_df = 1

max_features = 10 ** 5



vectorizer = CountVectorizer(stop_words=stop_words,

                            token_pattern=token_pattern,

                            ngram_range=ngram_range,

                            max_features=max_features)
X_train=vectorizer.fit_transform(df_train['text'])

X_test=vectorizer.transform(df_test['text'])
#抽出した特徴を出力

vocabulary = vectorizer.vocabulary_

print('id\ttoken')

print('-------------')

for token in sorted(vocabulary.keys(), key=lambda x: vocabulary[x]):

    print('{:d}\t{:s}'.format(vocabulary[token], token))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV, StratifiedKFold



parameters = {'penalty': ['l1', 'l2'],'C': [1.0, 3.0, 10.0, 30.0, 100.0, 300.0],'class_weight': [None, 'balanced']}



#乱数seed

random_state = 2000



X = X_train

y = df_train['target']





lgr = LogisticRegression(random_state=random_state, solver='liblinear')

clf = GridSearchCV(estimator=lgr, param_grid=parameters, scoring='f1',cv=4,iid=True)





clf.fit(X,y)
from sklearn.metrics import f1_score

train_predicted = clf.predict(X)

f1_score(train_predicted, y)
test_predicted = clf.predict(X_test)
sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

sub['target'] = list(map(int, test_predicted))

sub.to_csv('submission.csv', index=False)