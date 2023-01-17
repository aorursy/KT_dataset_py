import pandas as pd

import yake_helper_funcs as yhf

from sklearn.cluster import SpectralClustering

import numpy as np

import itertools



forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")



# get forum posts



# subsample forum posts

sample_posts = forum_posts.Message[-1000:].astype(str)
# extact keywords & tokenize

keywords = yhf.keywords_yake(sample_posts)

keywords_tokenized = yhf.tokenizing_after_YAKE(keywords)

keyword_sets = [set(post) for post in keywords_tokenized]



# remove empty sets

keyword_sets_noempty = [x for x in keyword_sets if x]
vectors = pd.read_csv("../input/fine-tuning-word2vec-2-0/kaggle_word2vec.model", 

                      delim_whitespace=True,

                      skiprows=[0], 

                      header=None

                     )



# set words as index rather than first column

vectors.index = vectors[0]

vectors.drop(0, axis=1, inplace=True)
def vectors_from_post(post):

    all_words = [] 



    for words in post:

        all_words.append(words) 

        

    return(vectors[vectors.index.isin(all_words)])





def doc_embed_from_post(post):

    test_vectors = vectors_from_post(post)



    return(test_vectors.mean())
# get document embeddings for 100 posts

num_of_posts = 100

doc_embeddings = np.zeros([num_of_posts, 300])





# TODO: handle posts where all words out OOV

for i in range(num_of_posts):

    embeddings = np.array(doc_embed_from_post(keyword_sets[i]))

    if np.isnan(embeddings).any():

        doc_embeddings[i,:] = np.zeros([1,300])

    else:

        doc_embeddings[i,:] = embeddings
# note that (default) k-means label assignment didn't work well

clustering = SpectralClustering(assign_labels="discretize").fit(doc_embeddings)
# look at the labels for each of the posts

clustering.labels_
# explore our posts by cluster

post_subset = keyword_sets[0:num_of_posts]



def get_keyword_set_by_cluster(number):

    cluster_index = list(clustering.labels_ == number)

    return(list(itertools.compress(post_subset, cluster_index)))
# this cluster looks like it's out of vocabular

get_keyword_set_by_cluster(3)
# this cluster looks like it's about deep learning

get_keyword_set_by_cluster(6)