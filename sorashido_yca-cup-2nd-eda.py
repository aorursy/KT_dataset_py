import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import spacy

import shap

from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split

from sklearn.inspection import permutation_importance

from lightgbm import LGBMClassifier

import pyLDAvis

import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
train = pd.read_csv("../input/ykc-2nd/train.csv")

test = pd.read_csv("../input/ykc-2nd/test.csv")

sub = pd.read_csv("../input/ykc-2nd/sample_submission.csv")

train.shape, test.shape, sub.shape
train.head()
test.head()
train.describe()
test.describe()
x = train["department_id"]

len(x.unique()) # 売り場は0-20で、21分類
# 前処理

x = train["product_name"]

x = x.apply(lambda words : words.lower().replace(",", "").replace("&", "").split(" "))

x = x.apply(lambda words : list(filter(lambda word: word != "", words))) # 'Pizza for One Suprema  Frozen Pizza' のように空白が2つ重なるケースを除去
import collections



def flatten(l):

    for el in l:

        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):

            yield from flatten(el)

        else:

            yield el



words = list(flatten(x))
# wordの出現回数

words_df = pd.DataFrame(nltk.FreqDist(words).most_common())

words_df.columns = ['keyword', 'count']
words_df.shape
# Top100を表示

words_df = words_df.head(100)

fig, axes = plt.subplots(ncols=1, figsize=(8, 20), dpi=100)

sns.barplot(y=words_df['keyword'], x=words_df['count'])
# lengthを取得

word_length = [len(w) for w in x]

sns.countplot(word_length, color='blue')
# https://spacy.io/

nlp = spacy.load("en_core_web_sm")



doc = nlp("this is a sentence.")

print(doc.vector, doc.vector.shape)
# t-SNE

x = train["product_name"]

y = train["department_id"]



x = x.apply(lambda words : words.lower().replace(",", "").replace("&", ""))



vecs = x.apply(lambda text: nlp(text).vector)

vec_df = pd.DataFrame(list(vecs))



svd = TSNE(n_components=2).fit_transform(vec_df)



fig, axes = plt.subplots(ncols=1, figsize=(12, 12))

sns.scatterplot(x=svd[:, 0], y=svd[:, 1], alpha=0.8, hue=y, palette="RdBu_r", legend="full")
# preprocessing

x = train["product_name"]



tf_vectorizer = CountVectorizer(stop_words='english', 

                                         max_features=40000, 

                                         lowercase = True,

                                         max_df = 0.8,

                                         min_df = 10)

dtm_tf = tf_vectorizer.fit_transform(x)

dtm_tf.shape
n_topics = 8

lda_tf = LatentDirichletAllocation(n_components=n_topics)

lda_tf.fit(dtm_tf)
pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
n_topics = 21

lda_tf = LatentDirichletAllocation(n_components=n_topics)

lda_tf.fit(dtm_tf)
pyLDAvis.sklearn.prepare(lda_tf, dtm_tf, tf_vectorizer)
train["svd_0"] = svd[:, 0]

train["svd_1"] = svd[:, 1]
train.head()
X = train[["order_rate", "order_dow_mode", "order_hour_of_day_mode", "svd_0", "svd_1"]]

y = train["department_id"]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = LGBMClassifier()

model.fit(train_X, train_y)



r = permutation_importance(model, val_X, val_y,

                            n_repeats=30,

                            random_state=0)
sorted_idx = r.importances_mean.argsort()

fig, ax = plt.subplots(figsize=(10, 4))

ax.boxplot(r.importances[sorted_idx].T,

           vert=False, labels=val_X.columns[sorted_idx])

ax.set_title("Permutation Importances")

fig.tight_layout()

plt.show()
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(val_X)
shap.summary_plot(shap_values, val_X, plot_type="bar")
shap.summary_plot(shap_values[0], val_X) # クラス0に対する散布図
# クラス0におけるshap valueと'svd_1'の関係

shap.dependence_plot("svd_1", shap_values[0], val_X)
shap.dependence_plot("order_rate", shap_values[0], val_X)