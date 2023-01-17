from brown_clustering_yangyuan import *

import pandas as pd

from nltk.tokenize import RegexpTokenizer
forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")
# take first 100 forum posts

sample_data = forum_posts.Message[:100].astype('str').tolist()



# toeknize

tokenizer = RegexpTokenizer(r'\w+')

sample_data_tokenized = [w.lower() for w in sample_data]

sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]
corpus = Corpus(sample_data_tokenized, 0.001)

clustering = BrownClustering(corpus, 6)

clustering.train()
clustering.get_similar('kaggle')
clustering.get_similar("error")
! pip install git+https://github.com/LIAAD/yake
import yake
# take keywords for each post & turn them into a text string "sentence"

simple_kwextractor = yake.KeywordExtractor()





# create empty list to save our "sentnecs" to

sentences = []



# subsample forum posts

sample_posts = forum_posts.Message[-1000:].astype(str)



for post in sample_posts:

    post_keywords = simple_kwextractor.extract_keywords(post)

    

    sentence_output = ""

    for word, number in post_keywords:

        sentence_output += word + " "

    

    sentences.append(sentence_output)





# use the sentences as input for brown clustering
tokenizer = RegexpTokenizer(r'\w+')

sample_data_tokenized = [w.lower() for w in sentences]

sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]
corpus = Corpus(sample_data_tokenized, 0.001)

clustering = BrownClustering(corpus, 3)

clustering.train()
# output is word + mutal information with provided word

clustering.get_similar('kaggle')
clustering.get_similar('error')
clustering.get_similar('deadline')
clustering.get_similar('submission')
len(clustering._codes['kaggle'])
clustering.helper.get_cluster(0)
# so it looks like all the clustres have been concatenated to a single array

# but they're in alphabetical order so we can use that to un-cat them 

megacluster = clustering.helper.get_cluster(0)



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
cluster_list[50:60]