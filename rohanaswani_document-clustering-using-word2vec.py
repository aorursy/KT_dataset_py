import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from gensim.models.word2vec import Word2Vec
reviewdata_df = pd.read_csv("reviewdata.csv",encoding="ansi")

training_dataset = reviewdata_df["Review"].values
tokenizer = RegexpTokenizer(r'\w+')
training_vector = [nltk.word_tokenize(single_review) for single_review in training_dataset[:5000]]
model = Word2Vec(training_vector, min_count=1,size=32,seed=123456)
model.save('reviews.model') 
model['great']
model.wv.n_similarity('bed','couch') 
model.wv.similarity('bed','couch')
model.most_similar("great")
