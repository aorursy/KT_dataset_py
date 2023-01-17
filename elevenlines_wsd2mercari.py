import nltk
import string
import re
import itertools
import numpy as np
import pandas as pd
import pickle
#import lda

from operator import itemgetter

from nltk.stem.porter import *
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn import tree

import warnings
warnings.filterwarnings('ignore')

PATH = "../input/"
nsamples = None
train_raw = pd.read_csv(f'{PATH}train.tsv', sep='\t', nrows=nsamples)
test_raw = pd.read_csv(f'{PATH}test_stg2.tsv', sep='\t', nrows=nsamples)
# sentence preprocessor
stop = set(stopwords.words('english'))
regex = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    
# ストップワードに入ってたりアルファベット以外で始まったり短かったりするやつはだめ
def check_word(w):
    return w not in stop and re.search('[a-zA-Z]', w) and len(w)>=3

# だめな単語を除去した文章を返す
def preprocess_sentence(text: str) -> str:
    text = regex.sub(" ", text) # remove punctuation
    tokens = filter(check_word, (w.lower() for w in text.split()))
    return " ".join(tokens)

# カテゴリ分割(使ってない)
def split_cat(text):
    try:
        return text.split("/")
    except:
        return ("", "", "")

# Preprocessor
# 次の２つを処理してそれぞれpreprocess_sentenceにかけ、"name", "text"にする
# - 名前・ブランド名(欠損値を""で埋める)を結合したもの
# - すべての自然言語でできた情報を結合したもの
# "category"はユニークな番号をふる
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["name"] = (df['name'].fillna('')
                  + ' ' + df['brand_name'].fillna('')).apply(preprocess_sentence)
    df["text"] = (df['name'].fillna('')
                  + ' '+ df['category_name'].fillna('')
                  + ' '+ df['item_description'].fillna('')).apply(preprocess_sentence)
    df["category"] = df["category_name"].astype("category").cat.codes
    return df[['name', 'text', 'shipping', 'item_condition_id', 'category']]
# DataFrameの特定の列を拾ってきて、第二引数以降で与えられる処理を行うパイプラインを作る関数
def on_field(field: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(field), validate=False), *vec)

# 引数に与えられた処理をした後、特異値分解で次元を削減する関数
def with_svd(*vec) -> Pipeline:
    return make_pipeline(*vec, TruncatedSVD(n_components=5))

tfidf_max_features = 100000

# 以下の処理をすべて行い、横に結合したベクトルを作るFeatureUnion
# - "name"を拾ってきてTF-IDFにかけ、次元削減
# - "text"を拾ってきてTF-IDFにかけ、次元削減
# - その他はそのまま値として出す
# 結果としては1サンプルあたり13次元のベクトルになる
# ついでに並列処理
vectorizer = make_union(
    on_field("name", with_svd(TfidfVectorizer(max_features=tfidf_max_features))),
    on_field("text", with_svd(TfidfVectorizer(max_features=tfidf_max_features,
                                              ngram_range=(1, 2)))),
    on_field(["shipping", "item_condition_id", "category"]),
    n_jobs=4)
train = preprocess(train_raw)
test = preprocess(test_raw)
combined = pd.concat([train, test])
vec = vectorizer.fit_transform(combined)
train_x, test_x= vec[:len(train.index)], vec[len(train.index):]
train_t = np.log(train_raw["price"] + 1)
# model = RandomForestRegressor(n_jobs=4, min_samples_leaf=5, n_estimators=200)
model = BaggingRegressor(tree.DecisionTreeRegressor(), n_estimators=100, max_samples=0.9)
model.fit(train_x, train_t)
model.score(train_x, train_t)
preds = model.predict(test_x)
preds = pd.Series(np.exp(preds) - 1)

submit = pd.concat([test_raw.test_id, preds], axis=1)
submit.columns = ['test_id', 'price']
submit.to_csv('submit_rf_base.csv', index=False)
