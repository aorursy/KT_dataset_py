# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
metadata_df = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
metadata_df.head()
abstract=pd.DataFrame(metadata_df['abstract'].copy())
abstract.head()
abstract.dropna(inplace=True)  # Removal of papers with a nul abstract
abstract=abstract.reset_index(drop=True)
import nltk
from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
stopwords_eng=stopwords.words('english')

lemm = WordNetLemmatizer()
# tokenizing data

tokenized_abstract=[]

for i in range(0,len(abstract['abstract'])):

    text_after_tokenization = word_tokenize(abstract['abstract'][i])

    tokenized_abstract.append(text_after_tokenization)
import string
processed_abstract=[]

for i in range(0,len(tokenized_abstract)):

    processed_text=[]

    for j in range(0,len(tokenized_abstract[i])):

        if ((tokenized_abstract[i][j] not in stopwords_eng) and (tokenized_abstract[i][j] not in string.punctuation)):

            processed_text.append(lemm.lemmatize(tokenized_abstract[i][j]))

            

    processed_abstract.append(processed_text)
# Creation of dictionary

import gensim

dictionary = gensim.corpora.Dictionary(processed_abstract)
# Displays the first 10 words in the dictionary

count=0

for k,v in dictionary.iteritems():

    print(str(k)+" "+str(v))

    count+=1

    if count>10:

        break
# Removal of words that occur in less than 10 documents, and in more than half of the documents , and in the remaining documents retain only the first 10000 most frequent tokens

dictionary.filter_extremes(no_below=10,no_above=0.5,keep_n=10000)
# we now take each document, find how many of these words occur in it, and each of their frequencies in the doc

bow_corpus = [dictionary.doc2bow(text) for text in processed_abstract]
# importing lda model

from gensim import models
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=4, id2word=dictionary, passes=2, workers=2)
# display the words that indicate a particular topic

for idx,topic in lda_model.print_topics(-1):

    print("Topic {}\nWords:{}".format(idx,topic))
# Visualization using t-SNE

from sklearn.manifold import TSNE

lda_corpus=lda_model[bow_corpus]
import gensim

matrix_lda_model = gensim.matutils.corpus2csc(lda_corpus)
array_lda_model = matrix_lda_model.T.toarray()
topic_weights = []

for row_list in array_lda_model:

    topic_weights.append(row_list)

topic_num = np.argmax(array_lda_model, axis=1)
tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')

tsne_lda = tsne_model.fit_transform(array_lda_model)

from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt

output_notebook()

n_topics = 4

mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

#mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])

plt.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1],color=mycolors[topic_num])

plt.show()