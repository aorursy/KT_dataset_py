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
import nltk



x_data_nps = []



for fileid in nltk.corpus.sentence_polarity.fileids(categories=['neg']):

    x_data_nps.extend([' '.join(sent) for sent in nltk.corpus.sentence_polarity.sents(fileid)])



y_data_nps = [0] * len(x_data_nps)



x_data_gut = []

for fileid in nltk.corpus.sentence_polarity.fileids(categories=['pos']):

    x_data_gut.extend([' '.join(sent) for sent in nltk.corpus.sentence_polarity.sents(fileid)])

    

y_data_gut = [1] * len(x_data_gut)



x_data_full = x_data_nps[:500] + x_data_gut[:500]

print(len(x_data_full))

y_data_full = y_data_nps[:500] + y_data_gut[:500]

print(len(y_data_full))
import numpy as np



x_data = np.array(x_data_full, dtype=object)

#x_data = np.array(x_data_full)

print(x_data.shape)

y_data = np.array(y_data_full)

print(y_data.shape)
train_indexes = np.random.rand(len(x_data)) < 0.80



print(len(train_indexes))

print(train_indexes[:10])
x_data_train = x_data[train_indexes]

y_data_train = y_data[train_indexes]



print(len(x_data_train))

print(len(y_data_train))
x_data_test = x_data[~train_indexes]

y_data_test = y_data[~train_indexes]



print(len(x_data_test))

print(len(y_data_test))
from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import wordnet



stopwords_list = stopwords.words('english')



lemmatizer = WordNetLemmatizer()



def my_tokenizer(doc):

    words = word_tokenize(doc)

    

    pos_tags = pos_tag(words)

    

    non_stopwords = [w for w in pos_tags if not w[0].lower() in stopwords_list]

    

    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    

    lemmas = []

    for w in non_punctuation:

        if w[1].startswith('J'):

            pos = wordnet.ADJ

        elif w[1].startswith('V'):

            pos = wordnet.VERB

        elif w[1].startswith('N'):

            pos = wordnet.NOUN

        elif w[1].startswith('R'):

            pos = wordnet.ADV

        else:

            pos = wordnet.NOUN

        

        lemmas.append(lemmatizer.lemmatize(w[0], pos))



    return lemmas
from sklearn.decomposition import TruncatedSVD



class SVDDimSelect(object):

    def fit(self, X, y=None):        

        try:

            self.svd_transformer = TruncatedSVD(n_components=round(X.shape[1]/2))

            self.svd_transformer.fit(X)

        

            cummulative_variance = 0.0

            k = 0

            for var in sorted(self.svd_transformer.explained_variance_ratio_)[::-1]:

                cummulative_variance += var

                if cummulative_variance >= 0.5:

                    break

                else:

                    k += 1

                

            self.svd_transformer = TruncatedSVD(n_components=k)

        except Exception as ex:

            print(ex)

            

        return self.svd_transformer.fit(X)

    

    def transform(self, X, Y=None):

        return self.svd_transformer.transform(X)

        

    def get_params(self, deep=True):

        return {}
from sklearn.pipeline import Pipeline

from sklearn import neighbors



clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('w2vec', Word2VecTransformer()),\

                       ('clf', clf)])
from sklearn.model_selection import RandomizedSearchCV

import scipy



par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}





hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
print(x_data_train)
hyperpar_selector.fit(X=x_data_train, y=y_data_train)
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
import pickle



string_obj = pickle.dumps(hyperpar_selector)
model_file = open('model.pkl', 'wb')



model_file.write(string_obj)



model_file.close()
model_file = open('model.pkl', 'rb')

model_content = model_file.read()



obj_classifier = pickle.loads(model_content)



model_file.close()



res = obj_classifier.predict(["clever but not especialy compelling"])



print(res)
from sklearn.metrics import accuracy_score

res = obj_classifier.predict(x_data_test)

print(accuracy_score(y_data_test, res))
from sklearn.metrics import classification_report

y_pred = obj_classifier.predict(x_data_test)



print(classification_report(y_data_test, y_pred, target_names=['pos','neg']))
from sklearn.pipeline import Pipeline

from sklearn import neighbors



clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('w2vec', Word2VecTransformer(2)),\

                       ('clf', clf)])
from sklearn.model_selection import RandomizedSearchCV

import scipy



par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}





hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
print(x_data_train)
hyperpar_selector.fit(X=x_data_train, y=y_data_train)
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
import pickle



string_obj = pickle.dumps(hyperpar_selector)
model_file = open('model.pkl', 'wb')



model_file.write(string_obj)



model_file.close()
model_file = open('model.pkl', 'rb')

model_content = model_file.read()



obj_classifier = pickle.loads(model_content)



model_file.close()



res = obj_classifier.predict(["clever but not especialy compelling"])



print(res)
from sklearn.metrics import accuracy_score

res = obj_classifier.predict(x_data_test)

print(accuracy_score(y_data_test, res))
from sklearn.metrics import classification_report

y_pred = obj_classifier.predict(x_data_test)



print(classification_report(y_data_test, y_pred, target_names=['pos','neg']))
import nltk



x_data_pos = []

for fileid in nltk.corpus.sentence_polarity.fileids():

    x_data_pos.extend([' '.join(sent) for sent in sentence_polarity.sents(categories=['pos'])])

    

y_data_pos = [1] * len(x_data_pos)



x_data_neg = []

for fileid in nltk.corpus.sentence_polarity.fileids():

    x_data_neg.extend([' '.join(sent) for sent in sentence_polarity.sents(categories=['neg'])])

    

y_data_neg = [0] * len(x_data_neg)



x_data_full = x_data_pos[:500] + x_data_neg[:500]

print(len(x_data_full))

y_data_full = y_data_pos[:500] + y_data_neg[:500]

print(len(y_data_full))
class GloveTransformer(object):

    

    def __init__(self, algo=2):

        self.algo = algo

    

    def fit(self, X, y=None):     

        X = [nltk.word_tokenize(x) for x in X]

        

        self.corpus = Corpus()

        self.corpus.fit(X, window=10)

        

        self.num_dim = 100       

        self.glove = Glove(no_components=self.num_dim, learning_rate=0.05)

        

        self.glove.fit(corpus.matrix, epochs=30, no_threads=1, verbose=True)

        self.glove.add_dictionary(self.corpus.dictionary)

        

        return self

    

    def transform(self, X, Y=None):        

        X = [nltk.word_tokenize(x) for x in X]

        

        return np.array([np.mean([self.glove.word_vectors[self.glove.dictionary[w]] for w in words if w in self.corpus.dictionary.keys()] or [np.zeros(self.num_dim)],axis=0) 

                         for words in X])



        

    def get_params(self, deep=True):

        return {}
import numpy as np



x_data = np.array(x_data_full, dtype=object)

#x_data = np.array(x_data_full)

print(x_data.shape)

y_data = np.array(y_data_full)

print(y_data.shape)
from sklearn.pipeline import Pipeline

from sklearn import neighbors



clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('gt', GloveTransformer()), \

                       ('clf', clf)])
from sklearn.model_selection import RandomizedSearchCV

import scipy



par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
print(x_data_train)
hyperpar_selector.fit(X=x_data_train, y=y_data_train)
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
from sklearn.metrics import *



y_pred = hyperpar_selector.predict(x_data_test)



print(accuracy_score(y_data_test, y_pred))