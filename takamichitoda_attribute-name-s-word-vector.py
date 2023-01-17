# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from tqdm import tqdm



fasttext_path = "../input/fasttext-english-word-vectors-including-subwords/wiki-news-300d-1M-subword.vec"



def load_vecs(word, *arr):

    return (word, np.asarray(arr, dtype='float32'))



vec_dic = dict(load_vecs(*line.rstrip().rsplit(' ')) for line in tqdm(open(fasttext_path)))
labels = pd.read_csv("../input/imet-2019-fgvc6/labels.csv")

labels.sample(5, random_state=42)
labels["type"] = labels["attribute_name"].map(lambda x: x.split("::")[0])

labels["name"] = labels["attribute_name"].map(lambda x: x.split("::")[1])

labels.sample(5, random_state=42)
tag_names = labels[labels["type"] == "tag"]["name"].values

culture_names = labels[labels["type"] == "culture"]["name"].values
def get_vec(w):

    try:

        return vec_dic[w]

    except KeyError:

        return np.zeros(300)

    

tag_vecs = []

for n in tag_names:

    vecs = [get_vec(w) for w in n.split()]

    vec = sum(vecs)/len(vecs)

    tag_vecs.append(vec)

tag_vecs = np.array(tag_vecs)



culture_vecs = []

for n in culture_names:

    vecs = [get_vec(w) for w in n.split()]

    vec = sum(vecs)/len(vecs)

    culture_vecs.append(vec)

culture_vecs = np.array(culture_vecs)
from sklearn.manifold import TSNE



tag_model = TSNE(n_components=2, random_state=42)

np.set_printoptions(suppress=True)

tag_model.fit_transform(tag_vecs)



culture_model = TSNE(n_components=2, random_state=42)

np.set_printoptions(suppress=True)

culture_model.fit_transform(culture_vecs)
import matplotlib.pyplot as plt 
plt.figure(figsize=(40,40))

plt.scatter(tag_model.embedding_[:, 0], tag_model.embedding_[:,1])



count = 0

for label, x, y in zip(tag_names, tag_model.embedding_[:, 0], tag_model.embedding_[:, 1]):

    count +=1

    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.show()
plt.figure(figsize=(40,40))

plt.scatter(culture_model.embedding_[:, 0], culture_model.embedding_[:,1])



count = 0

for label, x, y in zip(culture_names, culture_model.embedding_[:, 0], culture_model.embedding_[:, 1]):

    count +=1

    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.show()