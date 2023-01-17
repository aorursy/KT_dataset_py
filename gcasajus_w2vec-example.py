import itertools

from itertools import combinations, groupby

import collections

import numpy as np

import pandas as pd



# Embeddings

from gensim.models import Word2Vec

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
products = pd.read_csv("../input/products.csv")

departments = pd.read_csv("../input/departments.csv")

aisles = pd.read_csv("../input/aisles.csv")
prod_names = list(products['product_name'])

product_table = pd.DataFrame(prod_names, columns=['Products'])
# Make everything lowercase.

product_table['Products_mod'] = product_table['Products'].str.lower()



# Clean special characters.

product_table['Products_mod'] = product_table['Products_mod'].str.replace('\W', ' ')



# Split products into terms: Tokenize.

product_table['Products_mod'] = product_table['Products_mod'].str.split()

product_table.head()
# Add product and aisle information

enriched_prods = pd.merge(products, departments, on="department_id")

enriched_prods = pd.merge(enriched_prods, aisles, on="aisle_id")
enriched_prods[['product_id', 'product_name', 'department', 'aisle']]
# Append the tokenized column

product_table = pd.merge(enriched_prods[['product_name', 'department', 'aisle', 'department_id','aisle_id']], product_table, left_on="product_name", right_on="Products")

product_table.head()
w2vec_model = Word2Vec(list(product_table['Products_mod']), size=20, window=5, min_count=1, workers=4)
# Create  dictionaries to obtain product vectors



prod_word = dict()

for w in w2vec_model.wv.vocab:

    prod_word[w] = w2vec_model[w]
display(list(prod_word.items())[:2])
# VECTOR CALCULATION FOR PRODUCTS

# Cycle through each word in the product name to generate the vector.

prods_w2v = dict()

for index, row in product_table.iterrows():

    word_vector = list()

    #print(row['Products_mod'])

    for word in row['Products_mod']:

        word_vector.append(prod_word[word])

    

    prods_w2v[row['Products']] = np.average(word_vector, axis=0)
display(list(prods_w2v.items())[:2])
product_table['vectors'] = prods_w2v.values()
product_table.head()