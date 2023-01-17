# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os





# Any results you write to the current directory are saved as output.



import nltk

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import string

from nltk.corpus import wordnet









from sklearn import neighbors

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn import neighbors



from sklearn.model_selection import RandomizedSearchCV

import scipy





from sklearn.metrics import *

from sklearn.decomposition import TruncatedSVD

from gensim.models import Word2Vec

df = pd.read_csv('../input/corpus_categorias_treino.csv')
l = df['category'].unique()
l = df['category'].value_counts()

l 

df['words'][0]




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

        

        lemmas.append(lemmatizer.lemmatize(w[0].lower(), pos))



    return lemmas
t = []

for i in df['words']:

    t.append(my_tokenizer(i))



    
#t

print(my_tokenizer(df['words'][0]))

#frase_processada.append(' '.join(nova_frase))

frase_processada = list()

for i in t:

    frase_processada.append(' '.join(i))

df['words_tratada'] = frase_processada

df.category.replace(['adventure', 'government','religion','science_fiction'], [0,1,2,3], inplace=True)





frase_processada[0]


vetorizar = CountVectorizer(lowercase=False, max_features=50)

bag_of_words = vetorizar.fit_transform(df["words"])





treino, teste , classe_treino, classe_teste = train_test_split(bag_of_words, df["category"], random_state = 42)

clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')







clf.fit(treino, classe_treino)

previsao_teste  = clf.predict(teste)

acuracia = clf.score(teste, classe_teste)

print(acuracia)


#https://www.kaggle.com/sermakarevich/sklearn-pipelines-tutorial

clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('clf', clf)])



par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



#cross validation = cv

hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
hyperpar_selector.fit(X=treino, y=classe_treino)#teste, classe_teste
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))




y_pred = hyperpar_selector.predict(teste)

#treino, teste , classe_treino, classe_teste

print(accuracy_score(classe_teste, y_pred))
#aplicando LSA 
#tfidf_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer)



#tfs = tfidf_vectorizer.fit_transform(df["words_tratada"])



#svd_transformer = TruncatedSVD(n_components=1000)



#svd_transformer.fit(tfs)
#cummulative_variance = 0.0

#k = 0

#for var in sorted(svd_transformer.explained_variance_ratio_)[::-1]:

#    cummulative_variance += var

#    if cummulative_variance >= 0.5:

#        break

#    else:

#        k += 1

        

#print(k)
#svd_transformer = TruncatedSVD(n_components=k)

#svd_transformer.fit(tfs)

#svd_data = svd_transformer.transform(tfs)

#print(sorted(svd_transformer.explained_variance_ratio_)[::-1])
#treino, teste , classe_treino, classe_teste = train_test_split( svd_data, df["category"], random_state = 42)

#clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



#clf.fit(treino, classe_treino)

#previsao_teste  = clf.predict(teste)

#acuracia = clf.score(teste, classe_teste)

#print(acuracia)


#clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



#my_pipeline = Pipeline([('clf', clf)])


#par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



#cross validation = cv

#hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
#hyperpar_selector.fit(X=treino, y=classe_treino)#teste, classe_teste
#print("Best score: %0.3f" % hyperpar_selector.best_score_)

#print("Best parameters set:")

#best_parameters = hyperpar_selector.best_estimator_.get_params()

#for param_name in sorted(par.keys()):

#    print("\t%s: %r" % (param_name, best_parameters[param_name]))


#y_pred = hyperpar_selector.predict(teste)

#treino, teste , classe_treino, classe_teste

#print(accuracy_score(classe_teste, y_pred))




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




clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')

my_pipeline = Pipeline([('tfidf', TfidfVectorizer(tokenizer=my_tokenizer)),\

                       ('svd', SVDDimSelect()), \

                       ('clf', clf)])







par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



#cross validation = cv

hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
#treino, teste , classe_treino, classe_teste = train_test_split(df['words'], df['category'], random_state = 42)

treino, teste , classe_treino, classe_teste = train_test_split(df["words"], df['category'], random_state = 42)

hyperpar_selector.fit(X=treino, y=classe_treino)#teste, classe_teste
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))




y_pred = hyperpar_selector.predict(teste)

#treino, teste , classe_treino, classe_teste

print(accuracy_score(classe_teste, y_pred))
##Skip gram




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

frase_processada = list()

for i in t:

    frase_processada.append(' '.join(i))

df['words_tratada2'] = frase_processada
treino, teste , classe_treino, classe_teste = train_test_split( df["words"], df["category"], random_state = 42)
clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('w2vt', Word2VecTransformer()), ('clf', clf)])
par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



#cross validation = cv

hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
hyperpar_selector.fit(X=treino, y=classe_treino)
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
from sklearn.metrics import *



y_pred = hyperpar_selector.predict(teste)



print(accuracy_score(classe_teste, y_pred))
#cbow




#sg ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.





class Word2VecTransformer(object):

    

    ALGO_SKIP_GRAM=1

    ALGO_CBOW=2    

    

    def __init__(self, algo=2):    

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




clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



my_pipeline = Pipeline([('w2vt', Word2VecTransformer()), ('clf', clf)])




par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



#cross validation = cv

hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
hyperpar_selector.fit(X=treino, y=classe_treino)
print("Best score: %0.3f" % hyperpar_selector.best_score_)

print("Best parameters set:")

best_parameters = hyperpar_selector.best_estimator_.get_params()

for param_name in sorted(par.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))


y_pred = hyperpar_selector.predict(teste)



print(accuracy_score(classe_teste, y_pred))
#glove

#from gensim.scripts.glove2word2vec import glove2word2vec

#from glove import Corpus, Glove
#class GloveTransformer(object):

#    

#    def __init__(self, algo=2):

#        self.algo = algo

#    

#    def fit(self, X, y=None):     

#        X = [nltk.word_tokenize(x) for x in X]

#        

#        self.corpus = Corpus()

#        self.corpus.fit(X, window=10)

#        

#        self.num_dim = 100       

#        self.glove = Glove(no_components=self.num_dim, learning_rate=0.05)

#        

#        self.glove.fit(corpus.matrix, epochs=30, no_threads=1, verbose=True)

#        self.glove.add_dictionary(self.corpus.dictionary)

#        

#        return self

#    

#    def transform(self, X, Y=None):        

#        X = [nltk.word_tokenize(x) for x in X]

#        

#        return np.array([np.mean([self.glove.word_vectors[self.glove.dictionary[w]] for w in words if w in self.corpus.dictionary.keys()] or [np.zeros(self.num_dim)],axis=0) 

#                         for words in X])

#

#        

#    def get_params(self, deep=True):

#        return {}
#clf = neighbors.KNeighborsClassifier(n_neighbors=10, weights='uniform')



#my_pipeline = Pipeline([('gt', GloveTransformer()),('clf', clf)])
#par = {'clf__n_neighbors': range(1, 60), 'clf__weights': ['uniform', 'distance']}



#hyperpar_selector = RandomizedSearchCV(my_pipeline, par, cv=3, scoring='accuracy', n_jobs=1, n_iter=20)
#hyperpar_selector.fit(X=treino, y=classe_treino)
#print("Best score: %0.3f" % hyperpar_selector.best_score_)

#print("Best parameters set:")

#best_parameters = hyperpar_selector.best_estimator_.get_params()

#for param_name in sorted(par.keys()):

#    print("\t%s: %r" % (param_name, best_parameters[param_name]))
#y_pred = hyperpar_selector.predict(teste)



#print(accuracy_score(classe_teste, y_pred))