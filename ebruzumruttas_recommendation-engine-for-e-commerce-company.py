import pandas as pd

import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import math

import re

from scipy.sparse import csr_matrix

conn=sqlite3.connect(":memory:")
df=pd.read_csv('https://storage.googleapis.com/ty2020/reco.csv.gz');
pr=df[['productcontentid','ImageLink']].drop_duplicates('productcontentid')
pd.set_option('display.max_colwidth', -1)

df.head(5)
df.info()
category=pd.DataFrame(df.groupby('category_name').size())

category.rename(columns={0:'size'},inplace=True)

category.sort_values(by='size',ascending=False,inplace=True)

category.reset_index(inplace=True)

plt.figure(figsize=(20,10))

ax=sns.barplot(x='category_name',y='size',data=category.head(100))

plt.xticks(rotation=90)

plt.tight_layout()
bu=pd.DataFrame(df.groupby('business_unit').size())

bu.rename(columns={0:'size'},inplace=True)

bu.sort_values(by='size',ascending=False,inplace=True)

bu.reset_index(inplace=True)

plt.figure(figsize=(20,10))

ax=sns.barplot(x='business_unit',y='size',data=bu.head(100))

plt.xticks(rotation=90)

plt.tight_layout()
df[df.gender=='Unisex'].groupby('business_unit').size().sort_values(ascending=False).head(50)

df.gender=df.gender.fillna('Unisex')

gender=pd.DataFrame(df.groupby('gender').size())

gender.rename(columns={0:'size'},inplace=True)

gender.sort_values(by='size',ascending=False,inplace=True)

gender.reset_index(inplace=True)
plt.figure(figsize=(8,6))

ax=sns.barplot(x='gender',y='size',data=gender)

plt.xticks(rotation=90)

plt.tight_layout()
df.price.describe()
boxplot = df.boxplot(column=['price'])
df[df.price>1000].groupby('category_name').size().sum()
df_gender=df[['gender','productcontentid']]
df_gender.to_sql("df_gender",conn,if_exists='replace')
gend= pd.read_sql(

    """

    select gender, count(1) n

    from df_gender

    group by gender

    order by n desc

    """,conn

)
gend['g_weight']= gend[['n']].transform(lambda x: x/x.sum())
df1=df.merge(gend,on='gender').drop('n',axis=1)
scenario1=df.drop(['partition_date','orderparentid','user_id','color_id', 'gender','price'],axis=1)

import sqlite3

conn=sqlite3.connect(":memory:")
scenario1.to_sql("scenario1",conn,if_exists='replace')
bu_unit= pd.read_sql(

    """

    select business_unit, count(1) n

    from scenario1

    group by business_unit

    order by n desc

    """,conn

)
bu_unit['b_weight']= bu_unit[['n']].transform(lambda x: x/x.sum())

bu_unit=bu_unit.drop('n',axis=1)

df1=df1.merge(bu_unit,on=['business_unit'])
category=pd.read_sql(

    """

    select business_unit,category_name, count(1) n

    from scenario1

    group by  category_name ,business_unit

    order by n desc

    """

    ,conn)
category['c_weight']=category[['n']].transform(lambda x: x/x.sum())

category=category.drop('n',axis=1)

df1=df1.merge(category,on=['category_name','business_unit'])
df1['weight'] = df1.c_weight * 0.5 + df1.g_weight * 0.2 + df1.b_weight * 0.3

df1=df1.drop(['c_weight','g_weight','b_weight'],axis=1)
recommendation=df1.sort_values('weight',ascending=False).drop_duplicates(subset=['business_unit','gender'], keep="first").drop_duplicates(subset=['category_name','gender'],keep='first')
recommendation
from sklearn.cluster import KMeans

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel,cosine_similarity

from nltk.corpus import stopwords 

import nltk

nltk.download('stopwords')

df2=df1.copy()

df2=df2.sample(1000)
df2['description']=df2.business_unit.str.cat(" "+df2.category_name.str.cat(" "+df2.gender))
df2=df2[['productcontentid','description']]
stopw_turkish = stopwords.words('english') + stopwords.words('turkish')
tf = TfidfVectorizer(analyzer='word',stop_words='english')

tfidf_matrix=tf.fit_transform(df2.description)
cosine_similarities=linear_kernel(tfidf_matrix, tfidf_matrix)

df2=df2.reset_index().drop('index',axis=1)
results= {}
for idx, row in df2.iterrows():

    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]

    similar_items = [(cosine_similarities[idx][i], df2['productcontentid'][i]) for i in similar_indices]

    results[row['productcontentid']] = similar_items[1:]
def item(id):

    return df2.loc[df2['productcontentid'] == id]['description'].tolist()[0]
recommend(33178650,5)
!pip install surprise
from surprise import Reader, Dataset, SVD

from surprise import accuracy

from surprise.model_selection import train_test_split

from collections import defaultdict
df=df1.sample(5000)
reader=Reader(rating_scale=(0,1))

data=Dataset.load_from_df(df[['user_id','productcontentid','weight']],reader)

algo = SVD()

trainset = data.build_full_trainset()

algo.fit(trainset)

testset = trainset.build_anti_testset()

predictions = algo.test(testset)
def get_top_n(predictions, n=10):



    top_n = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:

        top_n[uid].append((iid, est))

        

    for uid, user_ratings in top_n.items():

        user_ratings.sort(key=lambda x: x[1], reverse=True)

        top_n[uid] = user_ratings[:n]



    return top_n
accuracy.rmse(predictions,verbose=True)
top_n = get_top_n(predictions, n=10)
top_n