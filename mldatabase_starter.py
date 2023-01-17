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
# Example

print("开始...")

import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer



df_train = pd.read_csv('../input/train_set.csv')

df_test = pd.read_csv('../input/test_set.csv')

df_train.drop(columns=['article', 'id'], inplace=True)

df_test.drop(columns=['article'], inplace=True)



vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)

vectorizer.fit(df_train['word_seg'])

x_train = vectorizer.transform(df_train['word_seg'])

x_test = vectorizer.transform(df_test['word_seg'])

y_train = df_train['class'] - 1



lg = LogisticRegression(C=4, dual=True)

lg.fit(x_train, y_train)

y_test = lg.predict(x_test)



df_test['class'] = y_test.tolist()

df_test['class'] = df_test['class'] + 1

df_result = df_test.loc[:, ['id', 'class']]

df_result.to_csv('result.csv', index=False)

print("完成...")