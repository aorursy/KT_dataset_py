import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import collections
from wordcloud import WordCloud
from gensim.models import Word2Vec
import os

print(os.listdir("../input"))
# input dataframe
df = pd.read_csv('../input/bhagavad-gita.csv')
df.head(5)
df.drop(df.columns[[0, 1, 3, 4]], axis=1, inplace=True)
df.head(10)
df = df.apply(lambda row: row['devanagari'].replace('।',' ').strip().split(), axis=1)
df.head(10)
data = []
for row in df:
    temp = []
    for words in row:
        if len(words)>3:
            temp.append(words)
    data.append(temp)
data_flat = [item for sublist in data for item in sublist]
# as per my knowledge of this holy document it makes sense to see these results because of the words used such as भारत, अर्जुन, कर्म, ज्ञानं.
top_10 = [i[0] for i in sorted(dict(collections.Counter(data_flat)).items(), key=lambda k: k[1], reverse=True)[:10]]
top_10
from nltk.text import Text  
bgita = Text(data_flat)
bgita.concordance('भारत')
def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

sim = 0.0
for i in data:
    for j in data:
        score = jaccard(set(i), set(j))
        if score > sim and i!=j:
            sim = score
            doc1 = i
            doc2 = j

print (round(sim*100, 2),'%', doc1, doc2)
model = Word2Vec(data, size=100, window=10, min_count=2, workers=4)
model.most_similar('अर्जुन')