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
# 訓練データとテストデータの読み込み

train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.describe   # "NaN"は欠損値
train.isnull().sum()   # 欠損値の確認
# locationの欠損値が多く、また予測に関係ないとみて削除

print(train.drop('location',axis=1))

print(test.drop('location',axis=1))
import matplotlib.pyplot as plt

import seaborn as sns
# ターゲットの要素とその個数をプロット

target_vals = train.target.value_counts()

sns.barplot(target_vals.index, target_vals)

plt.gca().set_ylabel('samples')
# 災害ツイートの例

disaster_tweets = train[train['target']==1]['text']

disaster_tweets.values[1]
# 非災害ツイートの例

non_disaster_tweets = train[train['target']==0]['text']

non_disaster_tweets.values[1]
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

print(stop_words)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing



# テキストデータの学習

# 出現した単語をカウントするシステムを構築

count_vectorizer = feature_extraction.text.CountVectorizer(stop_words=stop_words)



# ためしに最初の5つのツイートに対して実行

example_train_vectors = count_vectorizer.fit_transform(train["text"][0:5])
print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
# 全てのデータに対して実行

train_vectors = count_vectorizer.fit_transform(train["text"])

test_vectors = count_vectorizer.transform(test["text"])
# モデルの構築

model = linear_model.RidgeClassifier()
# trainデータを用いてモデルが機能するか確認

scores = model_selection.cross_val_score(model, train_vectors, train["target"], cv=3, scoring="f1")

scores
# 回帰分析の実行

result = model.fit(train_vectors, train["target"])
predictions = result.predict(test_vectors)
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = predictions
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)