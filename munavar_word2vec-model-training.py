import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gensim

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/train.csv")

df.head()
df.shape
def read_questions(row,column_name):

    return gensim.utils.simple_preprocess(str(row[column_name]).encode('utf-8'))

    

documents = []

for index, row in df.iterrows():

    documents.append(read_questions(row,"question1"))

    if row["is_duplicate"] == 0:

        documents.append(read_questions(row,"question2"))
print("List of lists. Let's confirm: ", type(documents), " of ", type(documents[0]))
model = gensim.models.Word2Vec(size=150, window=10, min_count=10, sg=1, workers=10)

model.build_vocab(documents)  # prepare the model vocabulary
model.train(sentences=documents, total_examples=len(documents), epochs=model.iter)
w1 = "phone"

model.wv.most_similar(positive=w1, topn=5)
w1 = ["women","rights"]

model.wv.most_similar (positive=w1,topn=2)
model.wv.doesnt_match(["government","corruption","peace"])