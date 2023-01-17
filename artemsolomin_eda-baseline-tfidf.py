# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc



from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer



from scipy import sparse



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/ocrv-intent-classification/train.csv', index_col='id')

df_train.head()
df_train.info()
df_train.describe()
df_train['label'].value_counts().plot(kind='bar');
df_test = pd.read_csv('/kaggle/input/ocrv-intent-classification/test.csv', index_col='id')

df_test.head()
df_test.info()
df_test.describe()
labels = df_train['label'].unique()

labels
vect_word = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',

                        stop_words= None,ngram_range=(1,3),dtype=np.float32)



vect_char = TfidfVectorizer(max_features=40000, lowercase=True, analyzer='char',

                        stop_words= None,ngram_range=(3,6),dtype=np.float32)
k = pd.DataFrame()

k['train'] = df_train.isnull().sum()

k['test'] = df_test.isnull().sum()

k
df_test[df_test['text'].isnull()]
df_test.fillna(' ',inplace=True)

tr_vect = vect_word.fit_transform(df_train['text'].apply(lambda tr_vect: np.str_(tr_vect)))

ts_vect = vect_word.transform(df_test['text'])



tr_vect_char = vect_char.fit_transform(df_train['text'].apply(lambda tr_vect: np.str_(tr_vect)))

ts_vect_char = vect_char.transform(df_test['text'])

X = sparse.hstack([tr_vect, tr_vect_char])

x_test = sparse.hstack([ts_vect, ts_vect_char])
y = df_train['label']
lr = LogisticRegression(C=23,n_jobs = 5, random_state = 123, solver='lbfgs', multi_class='multinomial', warm_start=True)

lr.fit(X,y)
y_pred = lr.predict(x_test)
y_pred.shape
type(y_pred)
df_test['label'] = np.array(y_pred)

df_test
df_test[['label']].to_csv('submission.csv')