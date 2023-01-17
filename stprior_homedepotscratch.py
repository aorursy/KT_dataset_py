import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.collocations import *
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn import model_selection


df = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
df["search_term"] = df["search_term"].str.lower()
df["product_title"] = df["product_title"].str.lower()


descr = pd.read_csv("../input/product_descriptions.csv",encoding="ISO-8859-1")
descr["product_description"] = descr["product_description"].str.lower()
df = df.merge(descr, on="product_uid")
df = df.assign(prod_complete = lambda x: (x['product_title'] + ' ' + x['product_description']))


s = df["relevance"]
notrelevant = df[s==1.00]
relevant = df[s==3.00]
import matplotlib.pyplot as plt
%matplotlib inline
s.hist()

#vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=1)
#search_counts = vectorizer.fit_transform(df["search_term"])
#distinct_title_counts = vectorizer.transform(df["product_title"].drop_duplicates())
#distinct_descr_counts = vectorizer.transform(
#    descr["product_description"].drop_duplicates())
#feature_counts = scipy.sparse.vstack(
#    [search_counts,distinct_title_counts,distinct_descr_counts])

notrelevant.sample(n=5)
nr1 = notrelevant[notrelevant["id"]==24058]
nr1

df[df["product_title"].str.contains("hydronic")]


prods =  pd.read_csv("../input/product_descriptions.csv", encoding="ISO-8859-1")
prods.sample(n=5)
hydronics = prods[prods["product_description"].str.contains("hydronic")]
hydronics[hydronics["product_description"].str.contains("heater")]

import nltk
from nltk.collocations import *
bm = nltk.collocations.BigramAssocMeasures()
#finder = BigramCollocationFinder.from_words(prods["product_description"].str.cat())

#score = finder.score_ngram(bm.pmi,"hydronic","heater")
#score


