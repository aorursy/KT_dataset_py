! pip install git+https://github.com/LIAAD/yake
from brown_clustering_yangyuan import *

import pandas as pd

from nltk.tokenize import RegexpTokenizer

import yake





forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")
def keywords_yake(sample_posts):

    # take keywords for each post & turn them into a text string "sentence"

    simple_kwextractor = yake.KeywordExtractor()



    # create empty list to save our "sentnecs" to

    sentences = []



    for post in sample_posts:

        post_keywords = simple_kwextractor.extract_keywords(post)



        sentence_output = ""

        for word, number in post_keywords:

            sentence_output += word + " "



        sentences.append(sentence_output)

        

    return(sentences)



def tokenizing_after_YAKE(sentences):

    tokenizer = RegexpTokenizer(r'\w+')

    sample_data_tokenized = [w.lower() for w in sentences]

    sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]

    

    return(sample_data_tokenized)



def get_clusters_maybe(megacluster):

    # so it looks like all the clustres have been concatenated to a single array

    # but they're in alphabetical order so we can use that to un-cat them 



    # create list with one sub list

    cluster_list = [[]] 

    list_index = 0



    # look at all words but last (since we compare each word

    # to the next word)

    for i in range(len(megacluster) - 1):

        if megacluster[i - 1] < megacluster[i]:

            # add current word to current sublist

            cluster_list[list_index].append(megacluster[i])

        else:

            # create a new sublist

            cluster_list.append([])

            list_index = list_index + 1



            # add current word

            cluster_list[list_index].append(megacluster[i])

    

    return(cluster_list)
# subsample forum posts

sample_posts = forum_posts.Message[-1000:].astype(str)



# extract keywords and token

keyword_output = keywords_yake(sample_posts)

tokenized_output = tokenizing_after_YAKE(keyword_output)



# cluster

corpus = Corpus(tokenized_output, 0.001)

clustering = BrownClustering(corpus, 2)

clustering.train()



# seperate output into different clusters

megacluster = clustering.helper.get_cluster(0)

clusters_maybe = get_clusters_maybe(megacluster)
clusters_maybe[10:20]