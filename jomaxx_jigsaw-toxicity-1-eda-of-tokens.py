# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud

import spacy

from spacy import displacy

from sklearn.decomposition import PCA



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
model = spacy.load("en_core_web_lg")
train_data.info()
comments = train_data[['comment_text','target']].copy()

comments_sample = comments.sample(n=1000)

comments_sample.head()
docs = []

for row in comments_sample.iterrows():

    doc = model(row[1][0])

    docs.append((doc, row[1][1], row[0]))
tokens = []



for (doc, target, _id) in docs:

    tokens.extend([(_id,\

                    target,\

                    token.text,\

                    token.lemma_,\

                    token.pos_,\

                    token.tag_,\

                    token.dep_,\

                    token.shape_,\

                    token.is_alpha,\

                    token.is_stop\

                   ) for token in doc])



tokens_df = pd.DataFrame(data = tokens, columns=['id','target','text','lemma','pos','tag','dep','shape','is_alpha', 'is_stop'])

print('Tokens Shape: {}'.format(tokens_df.shape))
tokens_df.sample(n=20)
noun_tokens_df = tokens_df[(tokens_df['pos'] == 'NOUN') & (tokens_df['is_alpha'] == True) & (tokens_df['is_stop'] == False)]

noun_tokens_df_counts = noun_tokens_df.groupby(by='lemma').agg({ 'target': 'mean', 'text': 'count'})
vectors = []

words = []

targets = []

counts = []





for doc, target, _id in docs:

    for token in doc:

        if token.has_vector and token.is_stop == False and token.pos_ == 'NOUN':

            product_count = tokens_df[tokens_df['text'] == token.text]['target'].unique().size            

            counts.append(product_count)

            targets.append(target)

            words.append(token.text)

            vectors.append(token.vector)

        

vec_df = pd.DataFrame(data = vectors, index=words)



pca = PCA(n_components=2)

red_vec = pca.fit_transform(vec_df.values)



vec_df['PC1'] = red_vec[:,0]

vec_df['PC2'] = red_vec[:,1]

vec_df['target'] = targets

vec_df['counts'] = counts

vec_df.head()
vec_df_pos = vec_df[vec_df['target'] >= 0.5]

plt.figure(figsize=(25,25))

p1 = sns.scatterplot(x='PC1', y='PC2', size='counts', hue='target', data=vec_df_pos)



for line in range(0,vec_df_pos.shape[0]):

    if vec_df_pos.counts[line] > 5: 

        p1.text(vec_df_pos.PC1[line]+0.2, vec_df_pos.PC2[line], vec_df_pos.index[line], horizontalalignment='left', size='large', color='black')
vec_df_neg = vec_df[vec_df['target'] < 0.5]

plt.figure(figsize=(25,25))

p1 = sns.scatterplot(x='PC1', y='PC2', size='counts', hue='target', data=vec_df_neg)



for line in range(0,vec_df_neg.shape[0]):

    if vec_df_neg.counts[line] > 5: 

        p1.text(vec_df_neg.PC1[line]+0.2, vec_df_neg.PC2[line], vec_df_neg.index[line], horizontalalignment='left', size='large', color='black')