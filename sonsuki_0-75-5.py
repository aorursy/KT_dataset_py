# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import DataFrame
term7=pd.read_csv("/kaggle/input/instacart-termdata/term7.csv")

term30=pd.read_csv("/kaggle/input/instacart-termdata/term30.csv")
term7.head()
from collections import Counter as cc



pro_name = cc(term7['product_name'])

A = pro_name.most_common()[:500]
A = DataFrame(A)

A.columns = ["product_name","product_count"]
A.head(10)
top7 = A.join(term7.set_index('product_name'), on='product_name')
print(top7)
top7['product_order'] =top7['product_id'] + top7['aisle_id'] * 0.001
top7.head(10)
top7.drop(['eval_set', 'product_count', 'product_id', 'aisle_id', 'days_since_prior_order'], axis='columns', inplace=True)

top7.head()
group7 = top7["user_id"].groupby(top7['product_order'])

group7 = pd.DataFrame(group7)
# -- 데이터프레임 출력 전체폭을 1000자로 확장

pd.set_option('display.width', 1000)



# -- 데이터프레임 출력 전체행을 1000개로 확장

# pd.set_option('display.height', 1000)



# -- 데이터프레임 컬럼 길이 출력 제약을 제거

pd.set_option('display.max_colwidth', -1)

 

pd.set_option('display.max_columns', None)



pd.set_option('display.max_rows', 500)
group7.head()
group7.columns = ['product_order','user_list']
group7.head()
type(group7.product_order)
# group7['user_list']=group7['user_list'].apply(str)

all_user_str = []

for i in range(0,len(group7.user_list)):

    user = group7.user_list[i].tolist()

    user_str = [str(s) for s in user]

#     user_str = list(set(user_str))

    all_user_str.append(user_str)
len(all_user_str[0])
all_user_str = []

for i in range(0,len(group7.user_list)):

    user = group7.user_list[i].tolist()

    user_str = [str(s) for s in user]

    user_str = list(set(user_str))

    all_user_str.append(user_str)
# user_str
# user = [list(s) for s in group7.user_list]
len(all_user_str[0])

# 확실히 수가 줄어듦을 확인, 그래도 어마어마하게 많이 묶여있었다
# len(group7.user_list)
print("data type is",type(all_user_str),"of",type(all_user_str[0]))
user = all_user_str

# 원래 짜둔 변수명으로 모델 넣기 전 바꿔주기
from gensim.models.word2vec import Word2Vec
%%time

# model = Word2Vec(user)

model = Word2Vec(user, size=100, window=5, min_count=1, workers=4)

model.save('term7_10000_user.model')
model.init_sims(replace=True)
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt



def tsne_plot(model):

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(128, 128)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
# tsne_plot(model)
from sklearn.cluster import DBSCAN

from gensim.models import Word2Vec



import pandas as pd

import re



import matplotlib.pyplot as plt

import matplotlib.font_manager as fm

# 워드 벡터를 클러스터링하기 위해서

word_vector = model.wv.vectors

word_vector

match_index = model.wv.index2word

model.init_sims(replace=True)
dbscan = DBSCAN(eps=0.75, min_samples=5)

clusters = dbscan.fit_predict(word_vector) # 워드 벡터를 클러스터링 # fit_predict 함수는 클러스터링 된 결과를 리스트로 산출해준다. 
df = pd.DataFrame(clusters, columns=["cluster"], index=match_index).reset_index()

df.columns = ["word", "cluster"]

print(df.head())
# 노이즈 포인트 제거

df = df[df["cluster"] != -1] #-1은 노이즈 포인트
print(df.groupby(["cluster"]).count())
min_cluster = df["cluster"].min()

max_cluster = df["cluster"].max()
print(min_cluster, max_cluster)
for df_num in range(min_cluster, max_cluster + 1):

    df_index = df[df["cluster"] == df_num].index

    df.loc[df_index, "value"] = list(range(0, len(df_index) * 3, 3))
df["cluster"].nunique()
grouped = df["word"].groupby(df["cluster"])

grouped = pd.DataFrame(grouped)

grouped.columns = ['cluster','word']

grouped
font = fm.FontProperties(size=70)

fig, ax = plt.subplots(figsize=(200, 189))

df.plot.scatter(x="cluster", y="value", ax=ax)

df[["cluster", "value", "word"]].apply(lambda x: ax.text(*x, fontproperties=font), axis=1)

plt.show()
# grouped.to_csv("term7_10000_dbscan.csv")