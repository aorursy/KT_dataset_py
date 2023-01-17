import pandas as pd

import numpy as np

from wordcloud import WordCloud,STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import matplotlib.pyplot as plt
reviews=pd.read_csv(r"../input/the-maze-runner-movie-review-data-set/The Maze Runner.csv",index_col=0)

reviews.head()
only_reviews = reviews.Reviews
all_reviews_text = " ".join(only_reviews)
all_reviews_text
all_reviews_text = all_reviews_text.lower()
wc=WordCloud(width=800,height=600,max_words=200,stopwords=STOPWORDS,regexp="[a-z']+")
plt.figure(figsize=(15,10))

plt.imshow(wc.generate_from_text(all_reviews_text))
tf = CountVectorizer(token_pattern="[a-z']+",stop_words=STOPWORDS,ngram_range=(2,2))
tf_mat =tf.fit_transform(reviews.Reviews)

tf_mat
d1={k:v for k, v in sorted(tf.vocabulary_.items(),key=lambda item: item[1])}
v=tf_mat.sum(axis=0) #we trying to get the total sum 
v1=v.tolist()[0]
len(v1)# number of columns
tf.vocabulary_ #the numbers is column index not the frequency
d2= {k1:v2 for k1,v2 in zip(d1.keys(),v1)}
plt.figure(figsize=(15,10))

plt.imshow(wc.generate_from_frequencies(d2))