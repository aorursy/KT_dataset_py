import gensim

from gensim.models import Word2Vec 

from gensim.models import KeyedVectors

import pandas as pd

from nltk.tokenize import RegexpTokenizer



forum_posts = pd.read_csv("../input/meta-kaggle/ForumMessages.csv")
# take first 100 forum posts

sample_data = forum_posts.Message[:100].astype('str').tolist()



# toeknize

tokenizer = RegexpTokenizer(r'\w+')

sample_data_tokenized = [w.lower() for w in sample_data]

sample_data_tokenized = [tokenizer.tokenize(i) for i in sample_data_tokenized]
model = KeyedVectors.load_word2vec_format("../input/word2vec-google/GoogleNews-vectors-negative300.bin",

                                         binary = True)
model.build_vocab(sample_data_tokenized, update=True)

model.train(sample_data_tokenized)