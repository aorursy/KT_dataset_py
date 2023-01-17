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
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
#負例と正例をひとつ表示
print(train_df[train_df["target"] == 0]["text"].values[1])
print(train_df[train_df["target"] == 1]["text"].values[1])
#sklearnの、Bag of Wordsで文章の特徴ベクトルを取るやつ
count_vectorizer = feature_extraction.text.CountVectorizer(stop_words='english')

#fit_transformという名前の関数は、sklearnではだいたい、「計算して処理する」もの。
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5]) #5行目まで

print(count_vectorizer.get_feature_names()) #どの単語がどこに対応しているか
print(example_train_vectors[0].todense().shape) #形状
print(example_train_vectors[0].todense())
#今度は全部学習する
#fit_transformにすると後でエラーになってしまう……？
#fit_transformと結果は変わらない？
train_vectors_f = count_vectorizer.fit_transform(train_df["text"])

test_vectors_f = count_vectorizer.fit_transform(test_df["text"])

train_vectors = count_vectorizer.transform(train_df["text"])

test_vectors = count_vectorizer.transform(test_df["text"])

print(train_vectors[0].todense().shape) #形状

clf = linear_model.RidgeClassifier()

#cross varidation
#cvの値が分割数
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=5, scoring="f1")
scores_f = model_selection.cross_val_score(clf, train_vectors_f, train_df["target"], cv=5, scoring="f1")

print(scores)
print(scores_f)
clf.fit(train_vectors, train_df["target"])
#https://www.kaggle.com/philculliton/nlp-getting-started-tutorialを参考にコピーしたもの
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
print(len(sample_submission))
print(len(sample_submission["target"]))
sample_submission

#なぜエラーになるのか
#ValueError: X has 12021 features per sample; expecting 21363

sample_submission["target"] = clf.predict(test_vectors) #predictで予測

#pred = clf.predict(test_vectors)
#sample_submission.head()
sample_submission

sample_submission.to_csv('submission_test.csv',index = False)