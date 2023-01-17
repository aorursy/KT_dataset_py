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
test_df = pd.read_csv('../input/test.csv')

# parse errorが発生したため、下記URLを参考にengine='python'を変数に追加

# https://www.shanelynn.ie/pandas-csv-error-error-tokenizing-data-c-error-eof-inside-string-starting-at-line/

train_df = pd.read_csv('../input/train.csv')
train_df.count()
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.ensemble import RandomForestClassifier

# BoWを作るための単語抽出器

vectorizer = CountVectorizer(min_df=2, stop_words='english')



all_content = list(train_df['question_text']) + list(test_df['question_text'])

train_content = list(train_df['question_text'])

test_content = list(test_df['question_text'])

# 訓練セット、テストセットすべてのBoW行列をベースとして作成

X_feat = vectorizer.fit(all_content)



# 訓練セットのBoWを作成

X_train = vectorizer.transform(train_content)

X_test = vectorizer.transform(test_content)

y_train = train_df['target']
# ランダムフォレスト

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(min_samples_leaf=3, random_state=0)

RF.fit(X_train, y_train)
y_test = (RF.predict_proba(X_test)[:, 0] < 0.78).astype(int)

sub = pd.read_csv('../input/sample_submission.csv')

sub['prediction'] = list(y_test)

sub.to_csv('submission.csv', index=False)