# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.
df = pd.read_csv('../input/train.csv', sep=',')

df = df.reset_index(drop=True)

df.head()
df.is_duplicate.mean()
df.shape
from sklearn.feature_extraction.text import CountVectorizer

df = df.dropna(how='any')

df.shape
BOW = CountVectorizer(max_df=0.999, min_df=1000, max_features=None, 

                                      analyzer='char', ngram_range=(1,2), 

                                      binary=True, lowercase=True)
%%time

BOW.fit(pd.concat((df.loc[:,'question1'],df.loc[:,'question2'])).unique())
%%time

BOW_1 = BOW.transform(df.loc[:,'question1'])

BOW_2 = BOW.transform(df.loc[:,'question2'])

labels = df.is_duplicate.values
X = -(BOW_1 != BOW_2).astype(int)

y = labels

X.shape, y.shape
%%time

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=0.1, solver='sag', class_weight={1: 0.46, 0: 1.32}).fit(X, y)
%%time

filepath = '../input/test.csv'

df = pd.read_csv(filepath)

df.loc[df['question1'].isnull(),['question1','question2']] = 'random empty question'

df.loc[df['question2'].isnull(),['question1','question2']] = 'random empty question'



BOW_1 = BOW.transform(df.loc[:,'question1'])

BOW_2 = BOW.transform(df.loc[:,'question2'])



X = -(BOW_1 != BOW_2).astype(int)
X.shape
%%time

scores = model.predict_proba(X)
df_out = pd.DataFrame()

df_out['test_id'] = df['test_id']

df_out['is_duplicate'] = scores[:,1]

df_out.to_csv('submission.csv', index=False)