import numpy as np

import pandas as pd
df = pd.read_csv("../input/cooking.csv")

df.head()
df["tags"] = df["tags"].map(lambda x: x.split())
print(df.loc[10])
print(df.iloc[10])
from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 
num_posts = df["content"].size

posts = []

for i in range( 0, num_posts ):

    posts.append( df["content"][i] )
train_data_features = vectorizer.fit_transform(posts)

train_data_features = train_data_features.toarray()



print(train_data_features.shape)
type(df.loc[0:5])
from gensim import corpora, models



dictionary = corpora.Dictionary(df.loc[0:5])