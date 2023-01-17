# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# we want some pretty loading bars

from tqdm._tqdm_notebook import tqdm_notebook

from tqdm import tqdm

# standard regex library

import re

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import codecs

from sklearn import feature_extraction

from nltk.stem.snowball import SnowballStemmer

import os

import matplotlib.pyplot as plt

import matplotlib as mpl

import spacy
# nlp = spacy.load('en_core_web_md')
# opening our two version

king_james_dir = "../input/old-and-new-english-bibles/king_james_bible.txt"

ams_dir = "../input/old-and-new-english-bibles/american standard version.csv"

books_dir = "/kaggle/input/bible-books/bible_books.txt"

ams = pd.read_csv(ams_dir)

american_standard_dir = "/kaggle/input/bible_books/american standard version.csv"

with open(king_james_dir, 'r') as file:

    king_james = file.read().replace('\n', '')
last_book = 1

last_chapter = 1

new = {}

current_text = ''

for i in tqdm(range(1, len(ams['t'])), desc="Preparing Data"):

    current_book = ams.at[i, 'b']

    current_chapter = ams.at[i, 'c']

    if last_book == current_book:

        current_text = current_text + ' ' + ams.at[i, 't']

    else:

        new[last_book] = current_text

        current_text = ams.at[i, 't']

        last_book = current_book

new[current_book] = current_text

ams = pd.DataFrame.from_dict(new, orient='index')

ams.columns = ['t']

file = open(books_dir, "r")

lines = file.readlines()

ams.index = lines
ams.info()
stemmer = SnowballStemmer("english")

# this is a customized tokenization function

# it not only tokenizes a string but also stems the tokens

# this saves an additional step when implemented into the TfidfVectorizer

def tokenize_and_stem(text):

    # tokenize by sentence

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []

    # filter punctuation and numbers

    for token in tokens:

        if re.search('[a-zA-Z]', token):

            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems
#    cv = CountVectorizer(ngram_range=(3,5), stop_words = 'english')

#    count_matrix = cv.fit_transform(ams.t)

#    count_matrix = (count_matrix.T * count_matrix)

 #   count_matrix.setdiag(0)

#    names = cv.get_feature_names()

#    to_gephi = pd.DataFrame(data = count_matrix.toarray(), columns = names, index = names)

#    to_gephi = to_gephi.loc[to_gephi.sum(axis=0) >= 15800 ,:]

#    cols = [x for x in to_gephi.columns if x not in to_gephi.index]

#    to_gephi.drop(inplace=True, axis = 1, labels=cols)

#    to_gephi.to_csv('occurences.csv', sep = ',')

# TfidfVectorizer which will transform texts into tfidf matrices

tfidf_vectorizer = TfidfVectorizer(max_df=0.75, max_features=12500,

                                 min_df=0.25, stop_words='english',

                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,6))

# Tfidf Matrix

tfidf_matrix = tfidf_vectorizer.fit_transform(ams['t'])

print("")
from sklearn.metrics.pairwise import cosine_similarity

distance_matrix = 1 - cosine_similarity(tfidf_matrix)
from sklearn.cluster import KMeans

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(tfidf_matrix)

clusters = kmeans.labels_.tolist()

ams['cluster'] = clusters

print(ams['cluster'].value_counts())
from sklearn.manifold import MDS

mds = MDS(n_components=2, verbose=1, n_jobs=-1)

pos = mds.fit_transform(distance_matrix)  # shape (n_components, n_samples)

x, y = pos[:, 0], pos[:, 1]
mds = MDS(n_components=3, verbose=1, n_jobs=-1)

pos3d = mds.fit_transform(distance_matrix)  # shape (n_components, n_samples)
from math import sqrt

def d_vector_diff(x, y):

    # take two 2D vectors as input and return the scalar distance between them

    return sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
# we search the point on our graph which is the most isolated

best = -500

ende = 0

for i in tqdm(range(len(x)), desc="searching..."):

    current = []

    for j in range(i, len(x)):

        current.append(d_vector_diff((x[i], y[i]),(x[j], y[j])))

    current_mean = np.mean(current)

    if best < current_mean:

        ende = (x[i], y[i])

        best = current_mean 
# we search for the two points on our graph which are furthes apart

best = -900

end = 0

for i in tqdm(range(len(x)), desc="searching..."):

    for j in range(i, len(x)):

        current = d_vector_diff((x[i], y[i]),(x[j], y[j]))

        if best < current:

            end = ((x[i], y[i]),(x[j], y[j]))

            best = current 
# this is just a very obnoxious way to append the texts to their appropiate place

americ_stand_ver = pd.read_csv(ams_dir)

texts = []

for i in tqdm(range(len(ams.index)), desc="Working..."):

    text = ''

    for j in range(len(americ_stand_ver['b'])):

        if i+1 == americ_stand_ver.at[j, 'b']:

            text = text + '' + americ_stand_ver.at[j, 't']

    texts.append(text)
# this will be needed later for plotting

plotting_frame = pd.DataFrame(dict(x=x, y=y, label=clusters, title=ams.index, text=texts))
# we form a new set of documents: Appending all texts of one cluster into a single one in which we can search for keywords

plotting_frame['by_label'] = ''

plotting_frame['keywords'] = ''

for i in tqdm(range(len(plotting_frame['text'])), desc= "Working..."):

    whole = ''

    for j in range(len(plotting_frame['text'])):

        if plotting_frame.at[i, 'label'] == plotting_frame.at[j, 'label']:

            whole = whole + ' ' + plotting_frame.at[j, 'text']

    plotting_frame.at[i, 'by_label'] = whole
# names and colors for visualization

cluster_names = {}

cluster_colors = {0: 'red', 1: 'purple', 2: 'black', 3: 'plum', 4: 'darksalmon', 5: 'gold', 6: 'indigo', 7: "yellow", 8: "green", 9: "red", 10:"blue"}



# quick import

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))

stop.add('seven')

stop.add('jesus christ')

stop.add('saith')

stop.add('ye say')

stop.add('christ')

stop.add('saith jehovah')







# for each cluster we extract its keywords to use them later as a label

for i in tqdm(range(len(plotting_frame['text'])), desc= "Keyword Extraction..."):

    # transform a single text

    transformed = tfidf_vectorizer.transform([plotting_frame.at[i, 'text']])

    # get the features from our vectorizer (which has been fit onto all texts)

    feature_array = np.array(tfidf_vectorizer.get_feature_names())

    # sort the transformed text by tfidf values

    tfidf_sorting = np.argsort(transformed.toarray()).flatten()[::-1]

    # takes top num values from the sorted transformed text

    num = 15

    top_n = feature_array[tfidf_sorting][:num]

        

    # append the keywords to the cluster they belong to

    cluster_names[plotting_frame.at[i, 'label']] = top_n
# group the frame by cluster

grouped = plotting_frame.groupby('label')
fig, ax = plt.subplots(figsize=(18, 18))

for name, gr in grouped:

        ax.plot(gr.x, gr.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')

ax.legend(numpoints=1)

for i in range(len(plotting_frame)):

    ax.set_xlim([-3,3])

    ax.set_ylim([-3,3])

    ax.text(plotting_frame.loc[i]['x'], plotting_frame.loc[i]['y'], plotting_frame.loc[i]['title'], size=8)

plt.show()
fig, ax = plt.subplots(figsize=(30, 7.5), nrows=1, ncols=len(cluster_names.keys()))

for c in cluster_names.keys():

    for name, gr in grouped:

        if name == c:

            ax[c].set_xlim([-3,3])

            ax[c].set_ylim([-3,3])

            ax[c].plot(gr.x, gr.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')

    ax[c].legend(numpoints=1)

    for i in range(len(plotting_frame)):

        if plotting_frame.at[i, 'label'] == c:

            ax[c].text(plotting_frame.loc[i]['x'], plotting_frame.loc[i]['y'], plotting_frame.loc[i]['title'], size=8)

plt.show()
fig, ax = plt.subplots(figsize=(18, 18))

for name, group in grouped:

        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')

ax.legend(numpoints=1)

ax.plot(end[0][0], end[0][1], color="magenta", marker="o", linestyle="", ms=19) 

ax.plot(end[1][0], end[1][1], color="magenta",marker="o", linestyle="", ms=19) 

ax.plot([end[1][0],end[0][0]], [end[1][1],end[0][1]], color="magenta", linestyle="-") 

for i in range(len(plotting_frame)):

    ax.text(plotting_frame.loc[i]['x'], plotting_frame.loc[i]['y'], plotting_frame.loc[i]['title'], size=8)

plt.show()
fig, ax = plt.subplots(figsize=(18, 18))

for name, group in grouped:

        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')

ax.legend(numpoints=1)

ax.plot(ende[0], ende[1], color="magenta",marker="o", linestyle="", ms=19) 

for i in range(len(plotting_frame)):

    ax.text(plotting_frame.loc[i]['x'], plotting_frame.loc[i]['y'], plotting_frame.loc[i]['title'], size=8)

plt.show()
# opening our two version

king_james_dir = "../input/old-and-new-english-bibles/king_james_bible.txt"

ams_dir = "../input/old-and-new-english-bibles/american standard version.csv"

books_dir = "/kaggle/input/bible-books/bible_books.txt"

ams = pd.read_csv(ams_dir)

american_standard_dir = "/kaggle/input/bible_books/american standard version.csv"

with open(king_james_dir, 'r') as file:

    king_james = file.read().replace('\n', '')
last_book = 1

last_chapter = 1

new = {}

current_text = ''

for i in tqdm(range(1, len(ams['t'])), desc="Preparing Data"):

    current_book = ams.at[i, 'b']

    current_chapter = ams.at[i, 'c']

    if last_chapter == current_chapter:

        current_text = current_text + ' ' + ams.at[i, 't']

    else:

        new[str(last_book) + '.' + str(last_chapter)] = current_text

        current_text = ams.at[i, 't']

        last_book = current_book

        last_chapter = current_chapter

new[str(current_book) + '.' + str(current_chapter)] = current_text

ams = pd.DataFrame.from_dict(new, orient='index')

ams.columns = ['t']
# TfidfVectorizer which will transform texts into tfidf matrices

tfidf_vectorizer = TfidfVectorizer(max_df=0.75, max_features=12500,

                                 min_df=0.25, stop_words='english',

                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,6))

# Tfidf Matrix

tfidf_matrix = tfidf_vectorizer.fit_transform(ams['t'])





distance_matrix = 1 - cosine_similarity(tfidf_matrix)



n_clusters = 4

kmeans = KMeans(n_clusters=n_clusters)

kmeans.fit(tfidf_matrix)

clusters = kmeans.labels_.tolist()

ams['cluster'] = clusters



mds = MDS(n_components=2, verbose=1, n_jobs=-1)

pos = mds.fit_transform(distance_matrix)  # shape (n_components, n_samples)

x, y = pos[:, 0], pos[:, 1]
# this is just a very obnoxious way to append the texts to their appropiate place

americ_stand_ver = pd.read_csv(ams_dir)

texts = []

for i in tqdm(range(len(ams.index)), desc="Working..."):

    text = ''

    for j in range(i ,len(americ_stand_ver['b'])):

        if str(ams.index[i]) == str(americ_stand_ver.at[j, 'b'])+'.'+ str(americ_stand_ver.at[j, 'c']):

            text = text + '' + americ_stand_ver.at[j, 't']

    texts.append(text)
# this will be needed later for plotting

plotting_frame = pd.DataFrame(dict(x=x, y=y, label=clusters, title=ams.index, text=texts))
# we form a new set of documents: Appending all texts of one cluster into a single one in which we can search for keywords

plotting_frame['by_label'] = ''

plotting_frame['keywords'] = ''

for i in tqdm(range(len(plotting_frame['text'])), desc= "Working..."):

    whole = ''

    for j in range(len(plotting_frame['text'])):

        if plotting_frame.at[i, 'label'] == plotting_frame.at[j, 'label']:

            whole = whole + ' ' + plotting_frame.at[j, 'text']

    plotting_frame.at[i, 'by_label'] = whole
# names and colors for visualization

cluster_names = {}

cluster_colors = {0: 'red', 1: 'purple', 2: 'black', 3: 'plum', 4: 'darksalmon', 5: 'gold', 6: 'indigo', 7: "yellow", 8: "green", 9: "red", 10:"blue"}



# for each cluster we extract its keywords to use them later as a label

for i in tqdm(range(len(plotting_frame['text'])), desc= "Keyword Extraction..."):

    # transform a single text

    transformed = tfidf_vectorizer.transform([plotting_frame.at[i, 'text']])

    # get the features from our vectorizer (which has been fit onto all texts)

    feature_array = np.array(tfidf_vectorizer.get_feature_names())

    # sort the transformed text by tfidf values

    tfidf_sorting = np.argsort(transformed.toarray()).flatten()[::-1]

    # takes top num values from the sorted transformed text

    num = 20

    top_n = feature_array[tfidf_sorting][:num]

    # append the keywords to the cluster they belong to

    cluster_names[plotting_frame.at[i, 'label']] = top_n
# group the frame by cluster

grouped = plotting_frame.groupby('label')
fig, ax = plt.subplots(figsize=(48, 48))

for name, gr in grouped:

     ax.plot(gr.x, gr.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')

ax.legend(numpoints=1)

for i in range(len(plotting_frame)):

    ax.text(plotting_frame.loc[i]['x'], plotting_frame.loc[i]['y'], plotting_frame.loc[i]['title'], size=8)

plt.show()
fig, ax = plt.subplots(figsize=(60, 15), nrows=1, ncols=len(cluster_names.keys()))

for c in cluster_names.keys():

    for name, gr in grouped:

        if name == c:

            ax[c].set_xlim([-9,9])

            ax[c].set_ylim([-9,9])

            ax[c].plot(gr.x, gr.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')

    ax[c].legend(numpoints=1)

    for i in range(len(plotting_frame)):

        if plotting_frame.at[i, 'label'] == c:

            ax[c].text(plotting_frame.loc[i]['x'], plotting_frame.loc[i]['y'], plotting_frame.loc[i]['title'], size=8)

plt.show()
 #   cv = CountVectorizer(ngram_range=(3,5), stop_words = 'english')

 #  count_matrix = cv.fit_transform(ams.t)

  #  count_matrix = (count_matrix.T * count_matrix)

 #   count_matrix.setdiag(0)
  #  names = cv.get_feature_names()

 #   to_gephi = pd.DataFrame(data = count_matrix.toarray(), columns = names, index = names)

#    to_gephi = to_gephi.loc[to_gephi.sum(axis=0) >= 1500 ,:]

 #   cols = [x for x in to_gephi.columns if x not in to_gephi.index]

 #   to_gephi.drop(inplace=True, axis = 1, labels=cols)

 #   to_gephi.to_csv('chapter_occurences.csv', sep = ',')