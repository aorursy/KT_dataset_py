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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#データの読み込み
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
#データの確認
train
#災害に関するTweetならtargetが'1'にそうでない場合は'0'
test
sub
#trainのデータを確認してみる
#ここでは、災害に関係しないTweetのテキスト(本文)を表示
train[train["target"] == 0]["text"]#.values[0]
#ここでは、災害に関係するTweetのテキスト(本文)を表示
train[train["target"] == 1]["text"]#.values[0]
#欠損値の確認

print(train.isnull().sum())
#欠損値の確認

print(test.isnull().sum())
#locationに欠損が多いことがわかる
#Tweetの傾向を確認

target_vals = train.target.value_counts()

#災害に関係ないTweetの数(train)

print(f'ターゲットの数"0" : {target_vals[0]}, "1" : {target_vals[1]}')
#可視化してみる

sns.barplot(target_vals.index, target_vals)

plt.gca().set_ylabel('Number')
#keywordの傾向を確認

print(f'keywordの単語数は？ :  "train" : {train["keyword"].nunique()}, "test" : {test["keyword"].nunique()}')
#keywordに出てくる単語数は221個？
#今後のために、textの単語数を調べる

print(f'textの単語数は？ :  "train" : {train["text"].nunique()}, "test" : {test["text"].nunique()}')
#keywordとtargetの関係を確認

#keywordとtargetの平均値を確認：1に近いほど、そのkeywordが災害と関係がある。一方、0に近いほど、関係がない。

ktm = train.groupby('keyword')['target'].mean()

ktm
pd.DataFrame(ktm)
ktm1 = ktm.sort_values()

ktm1
ktm1[ktm1>0.9]
ktm1[ktm1<0.1]
#図にしてみる

plt.figure(figsize = (20,10))

plt.plot(ktm1[ktm1>0.9].index,ktm1[ktm1>0.9])
#図にしてみる

plt.figure(figsize = (20,10))

plt.plot(ktm1[ktm1<0.1].index,ktm1[ktm1<0.1])
#テキストに出てくる単語数を特徴量とする、ベクトル化(向きが単語、長さが単語数)する

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

count_vectorizer = feature_extraction.text.CountVectorizer()
train_vectors = count_vectorizer.fit_transform(train["text"])

test_vectors = count_vectorizer.transform(test["text"])
#データ数確認

train_vectors
#データ数確認

test_vectors
#表形式で確認

pd.DataFrame(train_vectors.toarray(), columns = count_vectorizer.get_feature_names())
#表形式で確認

pd.DataFrame(test_vectors.toarray(), columns = count_vectorizer.get_feature_names())
#一般化線形モデルの適用

clf = linear_model.RidgeClassifier()
scores = model_selection.cross_val_score(clf, train_vectors, train["target"], cv=3, scoring="f1")

scores
#学習する

clf.fit(train_vectors, train["target"])
#予想する

sub["target"] = clf.predict(test_vectors)
#予想結果確認

sub.head()
#提出フォーマットへ

sub.to_csv("submission.csv", index = False)