import pandas as pd

import numpy as np

import os

import glob

import re # Regular Expressions

import pickle



# Natural Language Processing (NLP)

import nltk # Natural Language ToolKit

from gensim import corpora # Dictionaries

from nltk.stem import PorterStemmer # Porter stemming

from gensim.models import TfidfModel # TF-IDF models

from gensim import similarities # Similarity computations



# Clustering

from scipy.cluster import hierarchy



# Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

# sns.set(rc = {'figure.figsize': (30, 20)})

plt.rcParams["figure.figsize"] = (20,10)



# Display plots in a notebook

%matplotlib inline
# Apparently you may use different seed values at each stage

RANDOM_STATE = 2012



# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)



# 2. Set the `python` built-in pseudo-random generator at a fixed value

import random

random.seed(RANDOM_STATE)



# 3. Set the `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(RANDOM_STATE)
# Read .csv file with book information

books_df = pd.read_csv('../input/goodreads-best-books/book_data.csv')

books_df.head()
books_df.info()
# Draw a sample of books

books_df = (books_df.sample(replace = False, # Pick every row once

                         frac = 0.1, # 10% of books

                         random_state = RANDOM_STATE) # Random state for reproducibility

            .sort_index()

           .reset_index()) # Sort index
books_df.shape
HP_OP_BOOK = 'Harry Potter and the Order of the Phoenix'

HP_OP_INDEX = (books_df['book_title'] == HP_OP_BOOK).idxmax()
# What is the description for 'Harry Potter and the Order of the Phoenix'

books_df.iloc[HP_OP_INDEX]['book_desc']
# Remove all non-alpha-numeric characters

books_df['book_desc'] = books_df['book_desc'].str.replace(r'[\W_]+', ' ')
# Description after removing non-alpha-numeric characters

books_df.iloc[HP_OP_INDEX]['book_desc']
# Rows with missing descriptions

books_df[books_df['book_desc'].isnull()].head(n = 2)
# Leave rows with not missing descriptions

books_df = books_df[books_df['book_desc'].notnull()]

books_df.shape
# Rows with empty descirptions

books_df[books_df['book_desc'] == ' ']
# Remove rows with empty descriptions

books_df = books_df[-(books_df['book_desc'] == ' ')]

books_df.shape
# Split into tokens

books_df['book_desc_split'] = books_df['book_desc'].str.split()
# Count number of tokens

books_df['book_desc_len'] = books_df.apply(lambda row: len(row['book_desc_split']), 

                                        axis = 1) 
sns.distplot(books_df["book_desc_len"])

plt.show()
books_df[books_df["book_desc_len"] > 1100]
books_df[books_df["book_desc_len"] < 3]
# Creating a tokenizer

tokenizer = nltk.tokenize.RegexpTokenizer(pattern = '\w+')



# Tokenizing the text

books_df['tokens'] = books_df.apply(lambda row: tokenizer.tokenize(row['book_desc']), axis=1)



# Printing out the first 8 words / tokens 

books_df[['book_desc', 'tokens']].head()
# Convert the text to lower case 

books_df['tokens_lower'] = books_df.apply(lambda row: [token.lower() for token in row['tokens']] , 

                                          axis=1)



# Printing out words / tokens 

books_df[['book_desc', 'tokens', 'tokens_lower']].head()
# Getting the English stop words from nltk

stopwords = nltk.corpus.stopwords.words('english')

stopwords[1:10]
# Remove stopwords

books_df['tokens_no_sw'] = books_df.apply(lambda row: [token for token in row['tokens_lower'] 

                                                       if token not in stopwords] , 

                                          axis=1)



# Printing out the first 8 words / tokens 

books_df[['book_desc', 'tokens', 'tokens_lower', 'tokens_no_sw']].head()
# Create an instance of a PorterStemmer object

porter = PorterStemmer()



# Convert the text to lower case 

books_df['tokens_stem'] = books_df.apply(lambda row: [porter.stem(token) for token in row['tokens_no_sw']] , 

                                          axis=1)



# Printing out the first 8 words / tokens 

books_df[['book_desc', 'tokens', 'tokens_lower', 'tokens_no_sw', 'tokens_stem']].head()
# How many tokens?

books_df['tokens_len'] = books_df.apply(lambda row: len(row['tokens_stem']), 

                                        axis=1) 



# Printing out the first 8 words / tokens 

books_df[['book_desc', 'tokens', 'tokens_lower', 

          'tokens_no_sw', 'tokens_stem', 'tokens_len']].head()
sns.distplot(books_df["tokens_len"])

plt.show()
sns.distplot(books_df["book_desc_len"], label = 'Before')

sns.distplot(books_df["tokens_len"], label = 'After', color = 'red')

plt.legend()

plt.show()
# Create a dictionary from the stemmed tokens

dictionary = corpora.Dictionary(books_df['tokens_stem'])
# Create a bag-of-words model for each book, using the previously 

# generated dictionary

bow_model = [dictionary.doc2bow(book) for book in books_df['tokens_stem']]
# Convert the BoW model for Harry Potter and the Order of the Phoenix into a DataFrame

# For book nr

df_bow_HP_OP = pd.DataFrame(bow_model[HP_OP_INDEX])
# Add the column names to the DataFrame

df_bow_HP_OP.columns = ['index', 'occurrences']



# Add a column containing the token corresponding to the dictionary index

df_bow_HP_OP['token'] = df_bow_HP_OP['index'].apply(lambda x: dictionary[x])



# Sort the DataFrame by descending number of occurrences and 

# print the first 10 values

df_bow_HP_OP = df_bow_HP_OP.sort_values('occurrences', ascending=False)

df_bow_HP_OP.head(10)
# Generate the tf-idf model

tfidf_model = TfidfModel(bow_model)
# Convert the tf-idf model for Harry Potter and the Order of the Phoenix a DataFrame

df_tfidf = pd.DataFrame(tfidf_model[bow_model[HP_OP_INDEX]])



# Name the columns of the DataFrame id and score

df_tfidf.columns = ['id', 'score']



# Add the tokens corresponding to the numerical indices for better readability

df_tfidf['token'] = df_tfidf['id'].apply(lambda x: dictionary[x])



# Sort the DataFrame by descending tf-idf score and print the first 10 rows.

df_tfidf = df_tfidf.sort_values('score', ascending=False)

df_tfidf.head(10)
# Compute the similarity matrix (pairwise distance between all decriptions)

similar = similarities.MatrixSimilarity(tfidf_model[bow_model])
# Transform the resulting list into a dataframe

sim_df = pd.DataFrame(list(similar))



# Add the titles of the books as columns and index of the dataframe

sim_df.columns = books_df['book_title']

sim_df.index = books_df['book_title']
type(sim_df)
# Print the resulting matrix

sim_df.iloc[:3, :3]
# Select the column corresponding to selected book

similar_HP_OP = sim_df[HP_OP_BOOK]

similar_HP_OP.head()
# Sort by ascending scores

similar_HP_OP_sorted = similar_HP_OP.sort_values(ascending=False)

similar_HP_OP_sorted.head()
# Plot this data has a horizontal bar plot

similar_HP_OP_sorted[1:10].plot.barh(x='lab', y='val', rot=0).plot()



# Modify the axes labels and plot title for a better readability

plt.xlabel("Score")

plt.ylabel("Book")

plt.title("Similarity")
# Death Masks description

books_df[books_df['book_title'] == 'Death Masks'].iloc[0]['book_desc']
# White Night description

books_df[books_df['book_title'] == "White Night"].iloc[0]['book_desc']
# Compute the clusters from the similarity matrix,

# using the Ward variance minimization algorithm

clusters = hierarchy.linkage(sim_df, 'ward')
# Cut the hierarchy to make 100 clusters

cutree = hierarchy.cut_tree(clusters, 

                            n_clusters=100)
# Append cluster number to book_title

books_clust_df = pd.concat([books_df[['book_title', 'genres']].reset_index(drop = True), 

                                pd.DataFrame(cutree, columns = ['cluster'])], 

                          axis = 1)

books_clust_df.head()
# Filter books starting with 'Harry'

books_clust_df[books_clust_df['book_title'].str.contains('Harry')]
# Filter books that are in cluster that is the same as Harry Potter books

books_clust_df.query('cluster == 0')