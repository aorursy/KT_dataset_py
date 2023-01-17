import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import string

import re

%matplotlib inline
pre_df=pd.read_csv("/kaggle/input/flipkart-products/flipkart_com-ecommerce_sample.csv", na_values=["No rating available"])
pre_df.head()
pre_df.info()
pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('[]'))

pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.strip('"'))

pre_df['product_category_tree']=pre_df['product_category_tree'].map(lambda x:x.split('>>'))
#delete unwanted columns

del_list=['crawl_timestamp','product_url','image',"retail_price","discounted_price","is_FK_Advantage_product","product_rating","overall_rating","product_specifications"]

pre_df=pre_df.drop(del_list,axis=1)
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer 

lem = WordNetLemmatizer()

stop_words = set(stopwords.words('english')) 

exclude = set(string.punctuation)

import string
pre_df.head()
pre_df.shape
smd=pre_df.copy()

# drop duplicate produts

smd.drop_duplicates(subset ="product_name", 

                     keep = "first", inplace = True)

smd.shape
def filter_keywords(doc):

    doc=doc.lower()

    stop_free = " ".join([i for i in doc.split() if i not in stop_words])

    punc_free = "".join(ch for ch in stop_free if ch not in exclude)

    word_tokens = word_tokenize(punc_free)

    filtered_sentence = [(lem.lemmatize(w, "v")) for w in word_tokens]

    return filtered_sentence
smd['product'] = smd['product_name'].apply(filter_keywords)

smd['description'] = smd['description'].astype("str").apply(filter_keywords)

smd['brand'] = smd['brand'].astype("str").apply(filter_keywords)
smd["all_meta"]=smd['product']+smd['brand']+ pre_df['product_category_tree']+smd['description']

smd["all_meta"] = smd["all_meta"].apply(lambda x: ' '.join(x))
smd["all_meta"].head()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

# count_matrix = count.fit_transform(smd['all_meta'])

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')

tfidf_matrix = tf.fit_transform(smd['all_meta'])
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def get_recommendations(title):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:31]

    product_indices = [i[0] for i in sim_scores]

    return titles.iloc[product_indices]
smd = smd.reset_index()

titles = smd['product_name']

indices = pd.Series(smd.index, index=smd['product_name'])
get_recommendations("FabHomeDecor Fabric Double Sofa Bed").head(5)
get_recommendations("Alisha Solid Women's Cycling Shorts").head(5)
get_recommendations("Alisha Solid Women's Cycling Shorts").head(5).to_csv("Alisha Solid Women's Cycling Shorts recommendations",index=False,header=True)