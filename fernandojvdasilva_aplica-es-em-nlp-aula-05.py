import nltk

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string



hamlet_raw = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')

sentences = sent_tokenize(hamlet_raw)



all_words = [nltk.word_tokenize(sent) for sent in sentences]



for i in range(len(all_words)):  

    all_words[i] = [w.lower() for w in all_words[i] if w not in stopwords.words('english') and w not in string.punctuation]

    

print(all_words[:10])
from gensim.models import Word2Vec



#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.

word2vec = Word2Vec(all_words, min_count=2, sg=1)



vocabulary = word2vec.wv.vocab  

print(vocabulary)
v1 = word2vec.wv['heart']

print(v1)
sim_words = word2vec.wv.most_similar('truth')

print(sim_words)
!pip install glove_python
from glove import Corpus, Glove





corpus = Corpus() 



corpus.fit(all_words, window=10)



#creating a Glove object which will use the matrix created in the above lines to create embeddings

#We can set the learning rate as it uses Gradient Descent and number of components



glove = Glove(no_components=5, learning_rate=0.05)

 

glove.fit(corpus.matrix, epochs=30, no_threads=1, verbose=True)

glove.add_dictionary(corpus.dictionary)

glove.save('glove.model')



glove.add_dictionary(corpus.dictionary)
print(corpus.dictionary)
print(corpus.matrix)
# Vamos ver os vetores de uma palavra:

print(glove.word_vectors[glove.dictionary['heart']])

glove.most_similar('truth')
from gensim.models import Word2Vec



#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.





class Word2VecTransformer(object):

    

    ALGO_SKIP_GRAM=1

    ALGO_CBOW=2    

    

    def __init__(self, algo=1):

        self.algo = algo

    

    def fit(self, X, y=None):     

        X = [nltk.word_tokenize(x) for x in X]

        

        self.word2vec = Word2Vec(X, min_count=2, sg=self.algo)

        

        # Pegamos a dimensão da primeira palavra, para saber quantas dimensões estamos trabalhando,

        # assim podemos ajustar nos casos em que aparecerem palavras que não existirem no vocabulário.

        first_word = next(iter(self.word2vec.wv.vocab.keys()))

        self.num_dim = len(self.word2vec[first_word])       

        

        return self

    

    def transform(self, X, Y=None):        

        X = [nltk.word_tokenize(x) for x in X]

        

        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.num_dim)], axis=0) 

                         for words in X])



        

    def get_params(self, deep=True):

        return {}
