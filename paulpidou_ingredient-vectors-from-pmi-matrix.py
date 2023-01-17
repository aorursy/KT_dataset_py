%matplotlib inline



from collections import Counter

import itertools



import numpy as np

from scipy import sparse

from scipy.sparse import linalg 

from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import json

import sys

import os
recipes = []



with open(os.path.join('/kaggle/input/ingredient-sets', 'recipes.json'), 'r') as f:

    recipes = json.load(f)
recipes[0]
ingredients = [[ingredient['ingredientName'] for ingredient in recipe['ingredients']] for recipe in recipes]



# remove single ingredients recipes

ingredients = [ing for ing in ingredients if len(ingredients) > 1]

# show results

ingredients[0:5]
unigram_counts = Counter()

for i, ingredient in enumerate(ingredients):

    for token in ingredient:

        unigram_counts[token] += 1



tok2indx = {tok: indx for indx,tok in enumerate(unigram_counts.keys())}

indx2tok = {indx: tok for tok,indx in tok2indx.items()}



print('Done')

print('Vocabulary size: {}'.format(len(unigram_counts)))

print('Most common: {}'.format(unigram_counts.most_common(10)))
# Note we store the token vocab indices in the skipgram counter



skipgram_counts = Counter()

for ingredient in ingredients:

    tokens = [tok2indx[tok] for tok in ingredient]

    for i_ingredient, ingredient in enumerate(tokens):

        for i_context in range(len(tokens)):

            if i_ingredient == i_context:

                continue

            skipgram = (tokens[i_ingredient], tokens[i_context])

            skipgram_counts[skipgram] += 1    

        

print('Done')

print('Number of skipgrams: {}'.format(len(skipgram_counts)))

most_common = [

    (indx2tok[sg[0][0]], indx2tok[sg[0][1]], sg[1]) 

    for sg in skipgram_counts.most_common(10)]

print('Most common: {}'.format(most_common))
row_indxs = []

col_indxs = []

dat_values = []

i = 0

for (tok1, tok2), sg_count in skipgram_counts.items():

    i += 1  

    row_indxs.append(tok1)

    col_indxs.append(tok2)

    dat_values.append(sg_count)



iicnt_mat = sparse.csr_matrix((dat_values, (row_indxs, col_indxs)))

print('Done')
def ii_sim(ingredient, mat, topn=10):

    """Calculate topn most similar ingredients to ingredient"""

    indx = tok2indx[ingredient]

    if isinstance(mat, sparse.csr_matrix):

        v1 = mat.getrow(indx)

    else:

        v1 = mat[indx:indx+1, :]

    sims = cosine_similarity(mat, v1).flatten()

    sindxs = np.argsort(-sims)

    sim_ingredient_scores = [(indx2tok[sindx], sims[sindx]) for sindx in sindxs[0:topn]]

    return sim_ingredient_scores
ii_sim('beurre', iicnt_mat)
# Normalize each row using L2 norm

iicnt_norm_mat = normalize(iicnt_mat, norm='l2', axis=1)
# Demonstrate normalization

row = iicnt_mat.getrow(10).toarray().flatten()

print(np.sqrt((row*row).sum()))



row = iicnt_norm_mat.getrow(10).toarray().flatten()

print(np.sqrt((row*row).sum()))
ii_sim('poulet', iicnt_norm_mat)
num_skipgrams = iicnt_mat.sum()

assert(sum(skipgram_counts.values()) == num_skipgrams)



# for creating sparce matrices

row_indxs = []

col_indxs = []



pmi_dat_values = []    # pointwise mutual information

ppmi_dat_values = []   # positive pointwise mutial information

spmi_dat_values = []   # smoothed pointwise mutual information

sppmi_dat_values = []  # smoothed positive pointwise mutual information



# reusable quantities



# sum_over_rows[i] = sum_over_ingredients[i] = iicnt_mat.getcol(i).sum()

sum_over_ingredients = np.array(iicnt_mat.sum(axis=0)).flatten()

# sum_over_cols[i] = sum_over_contexts[i] = iicnt_mat.getrow(i).sum()

sum_over_contexts = np.array(iicnt_mat.sum(axis=1)).flatten()



# smoothing

alpha = 0.75

sum_over_ingredients_alpha = sum_over_ingredients**alpha

nca_denom = np.sum(sum_over_ingredients_alpha)



for (tok_ingredient, tok_context), sg_count in skipgram_counts.items():

    # here we have the following correspondance with Levy, Goldberg, Dagan

    #========================================================================

    #   num_skipgrams = |D|

    #   nwc = sg_count = #(w,c)

    #   Pwc = nwc / num_skipgrams = #(w,c) / |D|

    #   nw = sum_over_cols[tok_ingredient]    = sum_over_contexts[tok_ingredient] = #(w)

    #   Pw = nw / num_skipgrams = #(w) / |D|

    #   nc = sum_over_rows[tok_context] = sum_over_ingredients[tok_context] = #(c)

    #   Pc = nc / num_skipgrams = #(c) / |D|

    #

    #   nca = sum_over_rows[tok_context]^alpha = sum_over_ingredients[tok_context]^alpha = #(c)^alpha

    #   nca_denom = sum_{tok_content}( sum_over_ingredients[tok_content]^alpha )

    

    nwc = sg_count

    Pwc = nwc / num_skipgrams

    nw = sum_over_contexts[tok_ingredient]

    Pw = nw / num_skipgrams

    nc = sum_over_ingredients[tok_context]

    Pc = nc / num_skipgrams

    

    nca = sum_over_ingredients_alpha[tok_context]

    Pca = nca / nca_denom

    

    # note 

    # pmi = log {#(w,c) |D| / [#(w) #(c)]} 

    #     = log {nwc * num_skipgrams / [nw nc]}

    #     = log {P(w,c) / [P(w) P(c)]} 

    #     = log {Pwc / [Pw Pc]}

    pmi = np.log2(Pwc/(Pw*Pc))   

    ppmi = max(pmi, 0)

    spmi = np.log2(Pwc/(Pw*Pca))

    sppmi = max(spmi, 0)

    

    row_indxs.append(tok_ingredient)

    col_indxs.append(tok_context)

    pmi_dat_values.append(pmi)

    ppmi_dat_values.append(ppmi)

    spmi_dat_values.append(spmi)

    sppmi_dat_values.append(sppmi)

        

pmi_mat = sparse.csr_matrix((pmi_dat_values, (row_indxs, col_indxs)))

ppmi_mat = sparse.csr_matrix((ppmi_dat_values, (row_indxs, col_indxs)))

spmi_mat = sparse.csr_matrix((spmi_dat_values, (row_indxs, col_indxs)))

sppmi_mat = sparse.csr_matrix((sppmi_dat_values, (row_indxs, col_indxs)))



print('Done')
ii_sim('carotte', pmi_mat)
ii_sim('carotte', ppmi_mat)
ii_sim('carotte', spmi_mat)
ii_sim('carotte', sppmi_mat)
# Let's define a function to plot the vectors in a 2D space

def tsne_plot(word_to_vec_map):  

    labels, tokens = [], []



    for word, vector in word_to_vec_map.items():

        tokens.append(vector)

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x, y = [], []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(12, 12)) 

    for i in range(len(x)):

        plt.scatter(x[i], y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()
pmi_use = ppmi_mat

embedding_size = 30

uu, ss, vv = linalg.svds(pmi_use, embedding_size) 
print('Vocab size: {}'.format(len(unigram_counts)))

print('Embedding size: {}'.format(embedding_size))

print('uu.shape: {}'.format(uu.shape))

print('ss.shape: {}'.format(ss.shape))

print('vv.shape: {}'.format(vv.shape))
unorm = uu / np.sqrt(np.sum(uu*uu, axis=1, keepdims=True))

vnorm = vv / np.sqrt(np.sum(vv*vv, axis=0, keepdims=True))



ingredient_vecs = uu + vv.T

ingredient_vecs_norm = ingredient_vecs / np.sqrt(np.sum(ingredient_vecs*ingredient_vecs, axis=1, keepdims=True))
# Let's create a mapping from ingredient to vector

ingredient_to_vec_map = {}

for idx, vector in enumerate(ingredient_vecs_norm):

    ingredient_to_vec_map[indx2tok[idx]] = vector
tsne_plot(ingredient_to_vec_map)
def ingredient_sim(ingredient, sim_mat):

    sim_ingredient_scores = ii_sim(ingredient, sim_mat)

    for sim_ingredient, sim_score in sim_ingredient_scores:

        print(sim_ingredient, sim_score)
ingredient = 'carotte'

ingredient_sim(ingredient, ingredient_vecs)
def find_substitutes(ingredient, sim_mat, pmi_mat, threshold=None):

    substitutes = []

    

    candidates = ii_sim(ingredient, sim_mat, 20)

    indx1 = tok2indx[ingredient]

    for candidate, score in candidates:

        if candidate == ingredient:

            continue

        if threshold:

            if score < threshold:

                continue

                

        indx2 = tok2indx[candidate]

        pmi = pmi_mat[indx1, indx2]

        

        if pmi == 0: # We want ingredient that doesn't appear together within recipes

            substitutes.append((candidate, score))

    return substitutes
find_substitutes('tomate', iicnt_mat, ppmi_mat)
find_substitutes('tomate', ingredient_vecs_norm, ppmi_mat)
substitutes = {}

for ingredient in unigram_counts.keys():

    sub = find_substitutes(ingredient, iicnt_mat, ppmi_mat, 0.95)

    substitutes[ingredient] = sub
# Count the number of ingredients with possible substitutes

count = sum(1 for val in substitutes.values() if len(val) > 0)



# Ratio of ingredients covered

count / len(unigram_counts.keys())
# Retrieve ingredients and create translation map

ingredients_fr_en = {}

ingredients_en_fr = {}



with open(os.path.join('/kaggle/input/ingredient-sets', 'ingredients_fr_en.csv'), 'r') as f:

    for line in f:

        ing_fr, ing_en = line.split(',')

        ing_fr, ing_en = ing_fr.strip().lower(), ing_en.strip().lower()

            

        ingredients_fr_en[ing_fr] = ing_en

        ingredients_en_fr[ing_en] = ing_fr
# Is all ingredients in french map to single translation in english ?

len(ingredients_fr_en), len(ingredients_en_fr)
def get_word_to_vec_map(glove_file, words_to_keep): 

    word_to_vec_map = {}

    with open(glove_file, 'r', encoding='utf-8') as f: 

        for row in f:

            row = row.strip().split()

            word = row[0]

            if word in words_to_keep:

                word_to_vec_map[word] = np.array(row[1:], dtype=np.float64)

    return word_to_vec_map
def find_best_substitute(word, word_to_vec_map, ingredients_fr_en, ingredients_en_fr, threshold=None):

    word = word.lower()

    

    if not word in ingredients_fr_en.keys():

        return []

    word_en = ingredients_fr_en[word]

    

    if not word_en in word_to_vec_map:

        return []

    e_a = word_to_vec_map[word_en]

    

    max_cosine_sim = -sys.maxsize        # Initialize max_cosine_sim to a large negative number

    best_word = None                     # Initialize best_word with None, it will help keep track of the word to output



    for w in word_to_vec_map.keys():         

        # Compute cosine similarity between the vector e_a and the w's vector representation

        if w == word_en:

            continue

        cosine_sim = cosine_similarity([e_a], [word_to_vec_map[w]])[0][0]

        if threshold:

            if cosine_sim < threshold:

                continue

        

        if cosine_sim > max_cosine_sim:

            max_cosine_sim = cosine_sim

            best_word = w

     

    if best_word:

        return [(ingredients_en_fr[best_word], max_cosine_sim)]

    else:

        return []
word_to_vec_map = get_word_to_vec_map(os.path.join('/kaggle/input/glove840b300dtxt', 'glove.840B.300d.txt'), ingredients_en_fr.keys())
# Is all ingredients have a corresponding vectors within GloVe ?

len(word_to_vec_map)
tsne_plot(word_to_vec_map)
find_best_substitute('beurre', word_to_vec_map, ingredients_fr_en, ingredients_en_fr)
wv_substitutes = {}

for ingredient in unigram_counts.keys():

    sub = find_best_substitute(ingredient, word_to_vec_map, ingredients_fr_en, ingredients_en_fr, 0.7)

    wv_substitutes[ingredient] = sub
# Count the number of ingredients with possible substitutes

count = sum(1 for val in wv_substitutes.values() if len(val) > 0)



# Ratio of ingredients covered

count / len(unigram_counts.keys())