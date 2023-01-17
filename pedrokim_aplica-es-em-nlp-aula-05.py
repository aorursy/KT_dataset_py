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

import nltk
nltk.download('sentence_polarity')
from nltk.corpus import sentence_polarity

pos = sentence_polarity.sents(categories=['pos'])
neg = sentence_polarity.sents(categories=['neg'])

x_data_neg = []

for phrase in neg:
    x_data_neg.append(phrase)

y_data_neg = [0] * len(x_data_neg)

x_data_pos = []
for phrase in neg:
    x_data_pos.append(phrase)
    
y_data_pos = [1] * len(x_data_pos)

import numpy as np
from sklearn.model_selection import train_test_split

x_data_full = x_data_neg[:500] + x_data_pos[:500]
print(len(x_data_full))
y_data_full = y_data_neg[:500] + y_data_pos[:500]
print(len(y_data_full))


train_indexes = np.random.rand(len(x_data_full)) < 0.80

x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(x_data_full, y_data_full, test_size=0.2)

#type(x_data_train[0])
x_data_train
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import neighbors

clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')

my_pipeline = Pipeline([('w2v', Word2VecTransformer(algo=1)),\
                       ('clf', clf)])
from sklearn.model_selection import RandomizedSearchCV
import scipy

par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}

selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
selector.fit(X=x_data_train, y=y_data_train)
x_data_neg
for i in range(len(all_words)):  
    all_words[i] = [w.lower() for w in all_words[i] if w not in stopwords.words('english') and w not in string.punctuation]